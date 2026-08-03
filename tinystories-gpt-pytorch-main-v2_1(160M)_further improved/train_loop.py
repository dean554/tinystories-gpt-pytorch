print(f"Training on device: {device}")
train_losses, val_losses, steps_logged = [], [], []

# CHANGED: GradScaler is required for fp16 (not bf16). fp16 has a much
# narrower exponent range than fp32/bf16, so small gradient values can
# underflow to zero during backward and silently vanish. GradScaler
# multiplies the loss by a large factor before backward (pushing small
# gradients up into fp16's representable range), then unscales them
# before the optimizer step. It also auto-adjusts that scale factor over
# time and skips a step entirely if it detects inf/nan, so it's safe to
# just add it and not think about it further.
scaler = torch.amp.GradScaler('cuda')

@torch.no_grad()
def estimate_loss(eval_iters=10):
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        split_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            # CHANGED: bf16 -> fp16, since T4 (Turing, compute capability 7.5)
            # only has tensor-core support for fp16, not bf16.
            with torch.amp.autocast('cuda', dtype=torch.float16):
                _, loss = model(xb, yb)
            split_losses[k] = loss.mean().item()  # .mean() handles DataParallel's per-GPU loss vector
        losses[split] = split_losses.mean()
    model.train()
    return losses

eval_interval = 100
warmup_steps = 1200
max_steps = 20000
max_lr = 3e-4
min_lr = max_lr * 0.1

def get_lr(step):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

for i in range(start_step, max_steps + 1):
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    xb, yb = get_batch('train')

    # CHANGED: bf16 -> fp16 autocast.
    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits, loss = model(xb, yb)

    # NOTE: when wrapped in DataParallel, each GPU computes its own loss
    # on its slice of the batch, and these come back as a small vector
    # (one scalar per GPU) rather than a single number. .mean() reduces
    # that to one scalar so backward() has something valid to call on.
    loss = loss.mean()

    optimizer.zero_grad(set_to_none=True)

    # CHANGED: scaler.scale(loss).backward() instead of loss.backward().
    # This multiplies the loss by the current scale factor before running
    # backward, so the resulting gradients are scaled up proportionally.
    scaler.scale(loss).backward()

    # CHANGED: gradients must be unscaled before clipping, otherwise
    # clip_grad_norm_ would be clipping against the scaled (artificially
    # large) gradient values and cut off far more than max_norm=1.0 intends.
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # CHANGED: scaler.step(optimizer) replaces optimizer.step(). It checks
    # the unscaled gradients for inf/nan (which can happen if the scale
    # factor was too aggressive) — if found, it skips the update for this
    # step entirely rather than corrupting the weights. Otherwise it proceeds
    # as a normal optimizer.step().
    scaler.step(optimizer)

    # CHANGED: scaler.update() adjusts the scale factor for next time —
    # raising it if things have been stable for a while (to use more of
    # fp16's range), lowering it if it just saw an inf/nan.
    scaler.update()

    if i % 1000 == 0:
        # CHANGED: model.module.state_dict() instead of model.state_dict(),
        # since model is now a DataParallel wrapper — .module is the real
        # underlying GPTLanguageModel. This keeps checkpoint keys clean
        # (no "module." prefix), so they load directly into a plain
        # GPTLanguageModel later without needing to strip anything.
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        torch.save({
            'step': i,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"/kaggle/working/tiny_stories_checkpoint_step{i}.pt")

    if i % eval_interval == 0:
        losses = estimate_loss()
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        steps_logged.append(i)
        print(f"step {i}: train {losses['train']:.4f}, val {losses['val']:.4f}, lr {lr:.2e}")
