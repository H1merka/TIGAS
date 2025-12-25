"""
TIGAS Package - Full Test Suite
Полный тест всей функциональности с реальными изображениями

Запуск:
    python scripts/test_package_full.py
    python scripts/test_package_full.py --test_dir C:/path/to/test/images
    python scripts/test_package_full.py --skip-vis  # Пропустить тесты визуализации

Ожидаемая структура test_dir:
    test_dir/
        real/
            *.jpg, *.png
        fake/
            *.jpg, *.png
"""

import sys
import time
import argparse
import tempfile
from pathlib import Path

# Счётчики
passed = 0
failed = 0
skipped = 0
start_time = time.time()


def test(name: str, func, skip_condition=False, skip_reason=""):
    """Запуск теста с обработкой исключений."""
    global passed, failed, skipped
    if skip_condition:
        print(f"  ⊘ {name} (skipped: {skip_reason})")
        skipped += 1
        return
    try:
        func()
        print(f"  ✓ {name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        failed += 1


def main():
    global passed, failed, skipped
    
    parser = argparse.ArgumentParser(description="TIGAS Full Test Suite")
    parser.add_argument(
        "--test_dir",
        type=str,
        default=r"C:\Dev\TIGAS_dataset\TIGAS\test",
        help="Path to test dataset"
    )
    parser.add_argument(
        "--skip-vis",
        action="store_true",
        help="Skip visualization tests (no matplotlib required)"
    )
    args = parser.parse_args()
    
    test_dir = Path(args.test_dir)
    
    # Поддержка разных структур датасета:
    # 1. test_dir/real/ и test_dir/fake/
    # 2. test_dir/images/{generator}/0_real/ и 1_fake/
    real_dirs = []
    fake_dirs = []
    
    if (test_dir / "real").exists():
        real_dirs.append(test_dir / "real")
    if (test_dir / "fake").exists():
        fake_dirs.append(test_dir / "fake")
    
    # Поиск в структуре images/{generator}/0_real, 1_fake
    images_dir = test_dir / "images"
    if images_dir.exists():
        for generator_dir in images_dir.iterdir():
            if generator_dir.is_dir():
                real_sub = generator_dir / "0_real"
                fake_sub = generator_dir / "1_fake"
                if real_sub.exists():
                    real_dirs.append(real_sub)
                if fake_sub.exists():
                    fake_dirs.append(fake_sub)
    
    has_test_images = len(real_dirs) > 0 and len(fake_dirs) > 0
    
    print("=" * 60)
    print("TIGAS Package - Full Test Suite")
    print("=" * 60)
    print(f"Test directory: {test_dir}")
    print(f"Real image dirs found: {len(real_dirs)}")
    print(f"Fake image dirs found: {len(fake_dirs)}")
    print(f"Test images available: {has_test_images}")
    print(f"Skip visualization: {args.skip_vis}")
    
    import torch
    has_cuda = torch.cuda.is_available()
    print(f"CUDA available: {has_cuda}")
    if has_cuda:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # ==================== 1. Imports ====================
    print("\n[1/9] Testing imports...")
    
    def test_all_exports():
        import tigas
        expected = ['TIGAS', 'compute_tigas_score', 'load_tigas', 'TIGASMetric',
                    'get_default_model_path', 'download_default_model', 'clear_cache', 'cache_info']
        for name in expected:
            assert hasattr(tigas, name), f"Missing export: {name}"
    test("All public exports", test_all_exports)
    
    def test_submodule_imports():
        from tigas.models import TIGASModel, create_tigas_model
        from tigas.metrics import TIGASMetric
        from tigas.data import TIGASDataset
        from tigas.training import TIGASTrainer, CombinedLoss
        from tigas.utils import load_config, get_default_config
    test("Submodule imports", test_submodule_imports)
    
    # ==================== 2. Model Architecture ====================
    print("\n[2/9] Testing model architecture...")
    
    def test_model_params():
        from tigas.models.tigas_model import TIGASModel
        model = TIGASModel(img_size=256, fast_mode=True)
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total_params > 0
        assert trainable == total_params  # All should be trainable
        print(f"    Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    test("Model parameters", test_model_params)
    
    def test_model_size_info():
        from tigas.models.tigas_model import create_tigas_model
        model = create_tigas_model()
        info = model.get_model_size()
        assert 'total_parameters' in info
        assert 'model_size_mb' in info
    test("Model size info", test_model_size_info)
    
    def test_fast_vs_full_mode():
        from tigas.models.tigas_model import TIGASModel
        fast = TIGASModel(fast_mode=True)
        full = TIGASModel(fast_mode=False)
        fast_params = sum(p.numel() for p in fast.parameters())
        full_params = sum(p.numel() for p in full.parameters())
        assert full_params > fast_params, "Full mode should have more params"
        print(f"    Fast: {fast_params:,}, Full: {full_params:,}")
    test("Fast vs Full mode", test_fast_vs_full_mode)
    
    # ==================== 3. Forward Pass ====================
    print("\n[3/9] Testing forward pass...")
    
    def test_batch_sizes():
        from tigas.models.tigas_model import TIGASModel
        model = TIGASModel().eval()
        for bs in [1, 2, 4, 8]:
            x = torch.randn(bs, 3, 256, 256)
            with torch.no_grad():
                out = model(x)
            assert out['score'].shape == (bs, 1)
    test("Various batch sizes", test_batch_sizes)
    
    def test_different_img_sizes():
        from tigas.models.tigas_model import TIGASModel
        for size in [128, 256, 512]:
            model = TIGASModel(img_size=size).eval()
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                out = model(x)
            assert out['score'].shape == (1, 1)
    test("Different image sizes", test_different_img_sizes)
    
    def test_output_range():
        from tigas.models.tigas_model import TIGASModel
        model = TIGASModel().eval()
        for _ in range(10):
            x = torch.randn(4, 3, 256, 256)
            with torch.no_grad():
                scores = model(x)['score']
            assert (scores >= 0).all() and (scores <= 1).all()
    test("Output range [0, 1]", test_output_range)
    
    # ==================== 4. CUDA ====================
    print("\n[4/9] Testing CUDA...")
    
    def test_cuda_forward():
        from tigas.models.tigas_model import TIGASModel
        model = TIGASModel().eval().cuda()
        x = torch.randn(4, 3, 256, 256).cuda()
        with torch.no_grad():
            out = model(x)
        assert out['score'].device.type == 'cuda'
    test("CUDA forward pass", test_cuda_forward, skip_condition=not has_cuda, skip_reason="No CUDA")
    
    def test_cuda_memory():
        from tigas.models.tigas_model import TIGASModel
        torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        model = TIGASModel().eval().cuda()
        x = torch.randn(8, 3, 256, 256).cuda()
        with torch.no_grad():
            _ = model(x)
        after = torch.cuda.memory_allocated()
        mem_mb = (after - before) / 1024 / 1024
        print(f"    Memory used: {mem_mb:.1f} MB")
    test("CUDA memory usage", test_cuda_memory, skip_condition=not has_cuda, skip_reason="No CUDA")
    
    # ==================== 5. TIGAS API ====================
    print("\n[5/9] Testing TIGAS API...")
    
    def test_tigas_no_checkpoint():
        from tigas import TIGAS
        tigas = TIGAS(checkpoint_path=None, auto_download=False)
        x = torch.randn(1, 3, 256, 256)
        score = tigas(x)
        assert score.shape == (1, 1)
    test("TIGAS without checkpoint", test_tigas_no_checkpoint)
    
    def test_tigas_auto_download():
        from tigas import TIGAS
        tigas = TIGAS(auto_download=True, device='cpu')
        x = torch.randn(1, 3, 256, 256)
        score = tigas(x)
        assert score.shape == (1, 1)
    test("TIGAS with auto_download", test_tigas_auto_download)
    
    def test_compute_tigas_score():
        from tigas import compute_tigas_score
        x = torch.randn(1, 3, 256, 256)
        score = compute_tigas_score(x, auto_download=True)
        assert isinstance(score, (float, torch.Tensor))
    test("compute_tigas_score function", test_compute_tigas_score)
    
    # ==================== 6. Real Images ====================
    print("\n[6/9] Testing with real images...")
    
    def find_images(dirs, extensions=('*.jpg', '*.jpeg', '*.png', '*.JPEG')):
        """Найти изображения в списке директорий."""
        images = []
        for d in dirs:
            for ext in extensions:
                images.extend(d.glob(ext))
        return images
    
    if has_test_images:
        def test_single_image():
            from tigas import TIGAS
            tigas = TIGAS(auto_download=True, device='cuda' if has_cuda else 'cpu')
            real_images = find_images(real_dirs)
            if real_images:
                score = tigas(str(real_images[0]))
                print(f"    Real image score: {score.item():.4f}")
                assert 0 <= score.item() <= 1
            else:
                raise ValueError("No real images found")
        test("Single image inference", test_single_image)
        
        def test_directory_inference():
            from tigas import TIGAS
            tigas = TIGAS(auto_download=True, device='cuda' if has_cuda else 'cpu')
            # Берём первую директорию с real изображениями
            scores = tigas.compute_directory(str(real_dirs[0]), max_images=10)
            print(f"    Processed {len(scores)} images, mean score: {scores.mean():.4f}")
        test("Directory inference", test_directory_inference)
        
        def test_real_vs_fake():
            from tigas import TIGAS
            tigas = TIGAS(auto_download=True, device='cuda' if has_cuda else 'cpu')
            
            # Собираем изображения из всех директорий
            real_images = find_images(real_dirs)[:20]
            fake_images = find_images(fake_dirs)[:20]
            
            real_scores = []
            for img_path in real_images:
                score = tigas(str(img_path))
                real_scores.append(score.item())
            
            fake_scores = []
            for img_path in fake_images:
                score = tigas(str(img_path))
                fake_scores.append(score.item())
            
            real_mean = sum(real_scores) / len(real_scores) if real_scores else 0
            fake_mean = sum(fake_scores) / len(fake_scores) if fake_scores else 0
            
            print(f"    Real mean: {real_mean:.4f} ({len(real_scores)} images)")
            print(f"    Fake mean: {fake_mean:.4f} ({len(fake_scores)} images)")
            
            # Проверяем что модель различает real/fake (real должен быть выше)
            if real_mean > fake_mean:
                print(f"    ✓ Model correctly discriminates (diff: {real_mean - fake_mean:.4f})")
            else:
                print(f"    ⚠ Model discrimination weak (diff: {real_mean - fake_mean:.4f})")
        test("Real vs Fake discrimination", test_real_vs_fake)
        
        def test_pil_input():
            from tigas import TIGAS
            from PIL import Image
            tigas = TIGAS(auto_download=True, device='cpu')
            real_images = find_images(real_dirs)
            if real_images:
                img = Image.open(real_images[0]).convert('RGB')
                score = tigas(img)
                assert score.shape == (1, 1)
        test("PIL Image input", test_pil_input)
    else:
        print(f"  ⊘ Test images not found at {test_dir}")
        print(f"    Expected structure: test_dir/real/ and test_dir/fake/")
        print(f"    Or: test_dir/images/{{generator}}/0_real/ and 1_fake/")
        skipped += 4
    
    # ==================== 7. Model Hub ====================
    print("\n[7/9] Testing model hub...")
    
    def test_cache_info():
        from tigas import cache_info
        info = cache_info()
        assert isinstance(info, dict), f"cache_info should return dict, got {type(info)}"
        assert 'cache_dir' in info
        print(f"    Cache dir: {info['cache_dir']}")
    test("cache_info", test_cache_info)
    
    def test_get_default_model_path():
        from tigas import get_default_model_path
        path = get_default_model_path(auto_download=True)
        assert path is not None
        assert Path(path).exists()
    test("get_default_model_path", test_get_default_model_path)
    
    # ==================== 8. Visualization ====================
    print("\n[8/9] Testing visualization...")
    
    # Проверяем наличие matplotlib
    try:
        import matplotlib
        has_matplotlib = True
    except ImportError:
        has_matplotlib = False
    
    skip_vis = args.skip_vis or not has_matplotlib
    skip_vis_reason = "--skip-vis" if args.skip_vis else "matplotlib not installed (pip install tigas-metric[vis])"
    
    if not skip_vis:
        def test_matplotlib_available():
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        test("matplotlib import", test_matplotlib_available)
        
        def test_visualize_predictions():
            from tigas.utils.visualization import visualize_predictions
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            images = torch.randn(4, 3, 256, 256)
            # scores должен быть [B, 1], не [B]
            scores = torch.tensor([[0.9], [0.8], [0.3], [0.1]])
            labels = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
            visualize_predictions(images, scores, labels, save_path=temp_path)
            assert Path(temp_path).exists()
            plt.close('all')  # Close all figures to release file handles (Windows)
            Path(temp_path).unlink()  # Cleanup
        test("visualize_predictions", test_visualize_predictions)
        
        def test_plot_training_history():
            from tigas.utils.visualization import plot_training_history
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Функция ожидает два списка словарей: train_history и val_history
            train_history = [
                {'total': 0.5, 'regression': 0.3, 'classification': 0.2},
                {'total': 0.4, 'regression': 0.25, 'classification': 0.15},
                {'total': 0.3, 'regression': 0.2, 'classification': 0.1},
            ]
            val_history = [
                {'total': 0.6, 'regression': 0.35, 'classification': 0.25},
                {'total': 0.5, 'regression': 0.3, 'classification': 0.2},
                {'total': 0.4, 'regression': 0.25, 'classification': 0.15},
            ]
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
            plot_training_history(train_history, val_history, save_path=temp_path)
            assert Path(temp_path).exists()
            plt.close('all')  # Close all figures to release file handles (Windows)
            Path(temp_path).unlink()
        test("plot_training_history", test_plot_training_history)
    else:
        print(f"  ⊘ Visualization tests skipped ({skip_vis_reason})")
        skipped += 3
    
    # ==================== 9. Edge Cases ====================
    print("\n[9/9] Testing edge cases...")
    
    def test_single_image_batch():
        from tigas.models.tigas_model import TIGASModel
        model = TIGASModel().eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert out['score'].shape == (1, 1)
    test("Single image batch", test_single_image_batch)
    
    def test_normalized_input():
        from tigas import TIGAS
        tigas = TIGAS(auto_download=True, device='cpu')
        # Already normalized [-1, 1]
        x = torch.randn(1, 3, 256, 256).clamp(-1, 1)
        score = tigas(x)
        assert 0 <= score.item() <= 1
    test("Pre-normalized input", test_normalized_input)
    
    def test_unnormalized_input():
        from tigas import TIGAS
        tigas = TIGAS(auto_download=True, device='cpu')
        # [0, 1] range
        x = torch.rand(1, 3, 256, 256)
        score = tigas(x)
        assert 0 <= score.item() <= 1
    test("Unnormalized [0,1] input", test_unnormalized_input)
    
    def test_gradients():
        from tigas import TIGAS
        tigas = TIGAS(auto_download=True, device='cpu')
        tigas.model.train()
        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        score = tigas.model(x)['score']
        loss = (1 - score).mean()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    test("Gradient computation", test_gradients)
    
    # ==================== Summary ====================
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"Time: {elapsed:.1f}s")
    print("=" * 60)
    
    if failed > 0:
        print(f"\n✗ {failed} test(s) failed!")
        sys.exit(1)
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
