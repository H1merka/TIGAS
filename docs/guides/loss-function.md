# TIGAS как функция потерь для генеративных моделей

## Концепция

TIGAS полностью дифференцируем и может использоваться как функция потерь при обучении генеративных моделей (GAN, VAE, Diffusion и др.). Это позволяет напрямую оптимизировать генератор для создания более реалистичных изображений.

---

## Базовое использование

### Простая интеграция

```python
import torch
from tigas import TIGAS

# Инициализация
tigas = TIGAS(auto_download=True, device='cuda')
tigas.model.eval()  # Заморозить веса TIGAS

# В training loop генератора
def train_generator_step(generator, optimizer, noise):
    optimizer.zero_grad()
    
    # Генерация изображений
    fake_images = generator(noise)  # [B, 3, H, W], диапазон [0, 1]
    
    # TIGAS score как метрика качества
    # score=1.0 означает "реальное", score=0.0 означает "fake"
    tigas_score = tigas(fake_images)  # [B, 1]
    
    # Максимизируем score (минимизируем 1 - score)
    tigas_loss = 1.0 - tigas_score.mean()
    
    tigas_loss.backward()
    optimizer.step()
    
    return tigas_loss.item()
```

### С дополнительными потерями

```python
def train_generator_step(generator, discriminator, optimizer, noise, real_images):
    optimizer.zero_grad()
    
    fake_images = generator(noise)
    
    # Стандартная adversarial loss
    fake_logits = discriminator(fake_images)
    adversarial_loss = F.binary_cross_entropy_with_logits(
        fake_logits, 
        torch.ones_like(fake_logits)
    )
    
    # TIGAS loss как регуляризатор
    tigas_score = tigas(fake_images)
    tigas_loss = 1.0 - tigas_score.mean()
    
    # Комбинированная потеря
    total_loss = adversarial_loss + 0.1 * tigas_loss
    
    total_loss.backward()
    optimizer.step()
    
    return {
        'adversarial': adversarial_loss.item(),
        'tigas': tigas_loss.item()
    }
```

---

## Интеграция с популярными архитектурами

### GAN (Generative Adversarial Network)

```python
class TIGASGANTrainer:
    def __init__(self, generator, discriminator, tigas_weight=0.1):
        self.generator = generator
        self.discriminator = discriminator
        self.tigas = TIGAS(auto_download=True, device='cuda')
        self.tigas.model.eval()
        self.tigas_weight = tigas_weight
        
    def train_generator(self, noise, optimizer):
        optimizer.zero_grad()
        
        fake = self.generator(noise)
        
        # Adversarial loss
        d_fake = self.discriminator(fake)
        adv_loss = -torch.mean(d_fake)  # WGAN-style
        
        # TIGAS loss
        tigas_score = self.tigas(fake)
        tigas_loss = 1.0 - tigas_score.mean()
        
        # Total
        total = adv_loss + self.tigas_weight * tigas_loss
        total.backward()
        optimizer.step()
        
        return {'adv': adv_loss.item(), 'tigas': tigas_loss.item()}
    
    def train_discriminator(self, real, noise, optimizer):
        optimizer.zero_grad()
        
        with torch.no_grad():
            fake = self.generator(noise)
        
        d_real = self.discriminator(real)
        d_fake = self.discriminator(fake)
        
        # WGAN loss
        d_loss = torch.mean(d_fake) - torch.mean(d_real)
        
        d_loss.backward()
        optimizer.step()
        
        return {'d_loss': d_loss.item()}
```

### StyleGAN

```python
def stylegan_generator_loss(generator, mapping_net, w, tigas, tigas_weight=0.1):
    """
    StyleGAN generator loss с TIGAS регуляризацией.
    """
    # Генерация через mapping network и synthesis
    w_mapped = mapping_net(w)
    fake_images = generator.synthesis(w_mapped)
    
    # Path length regularization (standard StyleGAN)
    path_loss = path_length_regularization(generator, w_mapped)
    
    # TIGAS quality loss
    tigas_score = tigas(fake_images)
    tigas_loss = 1.0 - tigas_score.mean()
    
    return {
        'path_loss': path_loss,
        'tigas_loss': tigas_weight * tigas_loss,
        'total': path_loss + tigas_weight * tigas_loss
    }
```

### VAE (Variational Autoencoder)

```python
def vae_loss_with_tigas(vae, x, tigas, tigas_weight=0.1, kl_weight=1.0):
    """
    VAE loss с TIGAS компонентом.
    """
    # VAE forward
    recon, mu, logvar = vae(x)
    
    # Reconstruction loss
    recon_loss = F.mse_loss(recon, x)
    
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # TIGAS quality loss для реконструкций
    tigas_score = tigas(recon)
    tigas_loss = 1.0 - tigas_score.mean()
    
    total = recon_loss + kl_weight * kl_loss + tigas_weight * tigas_loss
    
    return {
        'recon': recon_loss.item(),
        'kl': kl_loss.item(),
        'tigas': tigas_loss.item(),
        'total': total
    }
```

### Diffusion Models

```python
def diffusion_denoising_with_tigas(
    denoiser, 
    noisy_images, 
    timesteps, 
    tigas,
    tigas_weight=0.05
):
    """
    Diffusion denoising loss с TIGAS регуляризацией.
    """
    # Предсказание шума
    predicted_noise = denoiser(noisy_images, timesteps)
    
    # Стандартная MSE loss
    noise_loss = F.mse_loss(predicted_noise, target_noise)
    
    # TIGAS loss (применяем к реконструированным изображениям)
    # Примечание: только для некоторых timesteps, чтобы не замедлять
    if timesteps[0] < 200:  # Близко к финальному изображению
        with torch.no_grad():
            reconstructed = reconstruct_from_noise(
                noisy_images, 
                predicted_noise, 
                timesteps
            )
        
        tigas_score = tigas(reconstructed)
        tigas_loss = 1.0 - tigas_score.mean()
        
        total = noise_loss + tigas_weight * tigas_loss
    else:
        total = noise_loss
        tigas_loss = torch.tensor(0.0)
    
    return total, {'noise': noise_loss.item(), 'tigas': tigas_loss.item()}
```

---

## Продвинутые техники

### Scheduled TIGAS Weight

```python
class TIGASScheduler:
    """
    Постепенное увеличение веса TIGAS loss.
    """
    def __init__(self, initial_weight=0.0, final_weight=0.1, warmup_epochs=10):
        self.initial = initial_weight
        self.final = final_weight
        self.warmup = warmup_epochs
        
    def get_weight(self, epoch):
        if epoch >= self.warmup:
            return self.final
        return self.initial + (self.final - self.initial) * (epoch / self.warmup)

# Использование
scheduler = TIGASScheduler(initial_weight=0.0, final_weight=0.1, warmup_epochs=10)

for epoch in range(100):
    tigas_weight = scheduler.get_weight(epoch)
    
    for batch in dataloader:
        loss = adv_loss + tigas_weight * tigas_loss
        # ...
```

### Gradient Clipping для TIGAS

```python
def train_step_with_clipping(generator, optimizer, noise, tigas, max_grad_norm=1.0):
    optimizer.zero_grad()
    
    fake = generator(noise)
    tigas_score = tigas(fake)
    loss = 1.0 - tigas_score.mean()
    
    loss.backward()
    
    # Clip gradients чтобы избежать взрывающихся градиентов от TIGAS
    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
    
    optimizer.step()
    return loss.item()
```

### Multi-Scale TIGAS

```python
def multi_scale_tigas_loss(generator, noise, tigas, scales=[1.0, 0.5, 0.25]):
    """
    TIGAS loss на нескольких масштабах.
    """
    fake = generator(noise)
    
    total_loss = 0.0
    for scale in scales:
        if scale != 1.0:
            scaled = F.interpolate(fake, scale_factor=scale, mode='bilinear')
        else:
            scaled = fake
        
        score = tigas(scaled)
        total_loss += (1.0 - score.mean()) / len(scales)
    
    return total_loss
```

---

## Мониторинг и отладка

### Логирование метрик

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/gan_with_tigas')

def train_epoch(generator, discriminator, dataloader, tigas, epoch):
    tigas_scores = []
    
    for i, (real, _) in enumerate(dataloader):
        noise = torch.randn(real.size(0), latent_dim, device=device)
        
        # Train discriminator
        d_loss = train_discriminator(...)
        
        # Train generator with TIGAS
        fake = generator(noise)
        tigas_score = tigas(fake)
        tigas_scores.append(tigas_score.mean().item())
        
        g_loss = -discriminator(fake).mean() + 0.1 * (1 - tigas_score.mean())
        # ...
    
    # Log average TIGAS score
    avg_tigas = sum(tigas_scores) / len(tigas_scores)
    writer.add_scalar('Generator/TIGAS_score', avg_tigas, epoch)
    
    print(f"Epoch {epoch}: Average TIGAS score = {avg_tigas:.4f}")
```

### Визуализация прогресса

```python
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def visualize_quality_progression(generator, tigas, fixed_noise, save_path):
    """
    Визуализация качества генерации с TIGAS score.
    """
    with torch.no_grad():
        fake = generator(fixed_noise)
        scores = tigas(fake)
    
    # Сортировка по score
    sorted_indices = scores.squeeze().argsort(descending=True)
    sorted_images = fake[sorted_indices]
    sorted_scores = scores[sorted_indices]
    
    # Grid с аннотациями
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i, (ax, img, score) in enumerate(zip(
        axes.flat, 
        sorted_images[:32], 
        sorted_scores[:32]
    )):
        img_np = (img.cpu().permute(1, 2, 0).numpy() + 1) / 2
        ax.imshow(img_np.clip(0, 1))
        ax.set_title(f'{score.item():.2f}', fontsize=8)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
```

---

## Лучшие практики

### Рекомендации по весам

| Модель | Рекомендуемый TIGAS weight | Примечание |
|--------|---------------------------|------------|
| GAN (начало обучения) | 0.0 - 0.01 | Дать дискриминатору научиться |
| GAN (основное обучение) | 0.05 - 0.1 | Баланс с adversarial loss |
| VAE | 0.1 - 0.2 | Помогает с reconstruction quality |
| Diffusion | 0.01 - 0.05 | Только на поздних timesteps |

### Чего избегать

1. **Слишком большой вес TIGAS:**
   ```python
   # ❌ Плохо - доминирует над основной задачей
   loss = adv_loss + 1.0 * tigas_loss
   
   # ✅ Хорошо - балансированное влияние
   loss = adv_loss + 0.1 * tigas_loss
   ```

2. **TIGAS на ранних этапах обучения:**
   ```python
   # ❌ Плохо - модель ещё не умеет генерировать
   for epoch in range(100):
       loss = adv_loss + 0.1 * tigas_loss
   
   # ✅ Хорошо - warmup период
   for epoch in range(100):
       tigas_weight = 0.1 if epoch > 10 else 0.0
       loss = adv_loss + tigas_weight * tigas_loss
   ```

3. **Обучение весов TIGAS:**
   ```python
   # ❌ Плохо - может деградировать
   tigas.model.train()
   
   # ✅ Хорошо - фиксированный претренированный detector
   tigas.model.eval()
   for param in tigas.model.parameters():
       param.requires_grad = False
   ```

---

## Диагностика проблем

| Симптом | Возможная причина | Решение |
|---------|-------------------|---------|
| TIGAS score не растёт | Слишком малый вес | Увеличить `tigas_weight` |
| Mode collapse | TIGAS доминирует | Уменьшить вес, добавить diversity loss |
| Генерация размытая | Конфликт с adversarial | Использовать perceptual loss |
| NaN в градиентах | Нестабильность TIGAS | Gradient clipping |
| Очень медленное обучение | TIGAS overhead | Вызывать реже (не каждый batch) |

### Оптимизация производительности

```python
# Вызывать TIGAS не на каждом шаге
def train_step(step, generator, noise, tigas, tigas_interval=10):
    fake = generator(noise)
    
    adv_loss = compute_adversarial_loss(fake)
    
    if step % tigas_interval == 0:
        tigas_score = tigas(fake)
        tigas_loss = 1.0 - tigas_score.mean()
    else:
        tigas_loss = torch.tensor(0.0, device=fake.device)
    
    total = adv_loss + 0.1 * tigas_loss
    return total
```
