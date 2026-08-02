print(f"Training on device: {device}")

train_losses, val_losses, steps_logged = [], [], []

@torch.no_grad()
def estimate_loss(eval_iters=10): #just a astimte
    model.eval()
    losses = {}
    for split in ['train', 'val']:
        split_losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            split_losses[k] = loss.item()
        losses[split] = split_losses.mean()
    model.train()
    return losses

eval_interval = 100

warmup_steps = 800
max_steps = 20000
max_lr = 1e-4
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
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if i % 500 == 0:
      torch.save({
          'step': i,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
      }, f"drive/MyDrive/checkpoints/checkpoint_step{i}.pt")

    if i % eval_interval == 0:
        losses = estimate_loss()
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        steps_logged.append(i)
        print(f"step {i}: train {losses['train']:.4f}, val {losses['val']:.4f}, lr {lr:.2e}")
