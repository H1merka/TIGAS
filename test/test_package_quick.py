"""
TIGAS Package - Quick Smoke Test
Быстрый тест базовой функциональности (~10-30 секунд)
Использует синтетические данные (random tensors)

Запуск:
    python scripts/test_package_quick.py
"""

import sys
import time

# Счётчики
passed = 0
failed = 0
start_time = time.time()


def test(name: str, func):
    """Запуск теста с обработкой исключений."""
    global passed, failed
    try:
        func()
        print(f"  ✓ {name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        failed += 1


def main():
    global passed, failed
    
    print("=" * 60)
    print("TIGAS Package - Quick Smoke Test")
    print("=" * 60)
    
    # ==================== 1. Imports ====================
    print("\n[1/6] Testing imports...")
    
    def test_main_imports():
        from tigas import TIGAS, compute_tigas_score, load_tigas
        assert TIGAS is not None
        assert compute_tigas_score is not None
        assert load_tigas is not None
    test("Main API imports", test_main_imports)
    
    def test_metric_import():
        from tigas import TIGASMetric
        assert TIGASMetric is not None
    test("TIGASMetric import", test_metric_import)
    
    def test_model_hub_imports():
        from tigas import get_default_model_path, download_default_model, clear_cache, cache_info
        assert get_default_model_path is not None
    test("Model hub imports", test_model_hub_imports)
    
    def test_version():
        import tigas
        assert hasattr(tigas, '__version__')
        assert isinstance(tigas.__version__, str)
    test("Version attribute", test_version)
    
    # ==================== 2. Model Creation ====================
    print("\n[2/6] Testing model creation...")
    
    def test_model_creation():
        from tigas.models.tigas_model import TIGASModel
        model = TIGASModel(img_size=256, fast_mode=True)
        assert model is not None
        assert model.img_size == 256
    test("TIGASModel creation (fast_mode)", test_model_creation)
    
    def test_model_full_mode():
        from tigas.models.tigas_model import TIGASModel
        model = TIGASModel(img_size=256, fast_mode=False)
        assert model is not None
    test("TIGASModel creation (full_mode)", test_model_full_mode)
    
    def test_create_tigas_model():
        from tigas.models.tigas_model import create_tigas_model
        model = create_tigas_model(img_size=256)
        assert model is not None
    test("create_tigas_model factory", test_create_tigas_model)
    
    # ==================== 3. Forward Pass (CPU) ====================
    print("\n[3/6] Testing forward pass (CPU)...")
    
    import torch
    
    def test_forward_cpu():
        from tigas.models.tigas_model import TIGASModel
        model = TIGASModel(img_size=256).eval()
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            out = model(x)
        assert 'score' in out
        assert 'logits' in out
        assert out['score'].shape == (2, 1)
        assert out['logits'].shape == (2, 2)
        assert (out['score'] >= 0).all() and (out['score'] <= 1).all()
    test("Forward pass CPU", test_forward_cpu)
    
    def test_forward_with_features():
        from tigas.models.tigas_model import TIGASModel
        model = TIGASModel(img_size=256).eval()
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = model(x, return_features=True)
        assert 'features' in out
    test("Forward pass with features", test_forward_with_features)
    
    # ==================== 4. Forward Pass (CUDA) ====================
    print("\n[4/6] Testing forward pass (CUDA)...")
    
    if torch.cuda.is_available():
        def test_forward_cuda():
            from tigas.models.tigas_model import TIGASModel
            model = TIGASModel(img_size=256).eval().cuda()
            x = torch.randn(2, 3, 256, 256).cuda()
            with torch.no_grad():
                out = model(x)
            assert out['score'].device.type == 'cuda'
        test("Forward pass CUDA", test_forward_cuda)
        
        def test_cuda_cpu_consistency():
            from tigas.models.tigas_model import TIGASModel
            torch.manual_seed(42)
            model = TIGASModel(img_size=256).eval()
            x = torch.randn(1, 3, 256, 256)
            with torch.no_grad():
                out_cpu = model(x)['score'].item()
                model_cuda = model.cuda()
                out_cuda = model_cuda(x.cuda())['score'].cpu().item()
            assert abs(out_cpu - out_cuda) < 0.01, f"CPU={out_cpu}, CUDA={out_cuda}"
        test("CUDA/CPU consistency", test_cuda_cpu_consistency)
    else:
        print("  ⊘ CUDA not available, skipping GPU tests")
    
    # ==================== 5. TIGASMetric ====================
    print("\n[5/6] Testing TIGASMetric...")
    
    def test_tigas_metric():
        from tigas import TIGASMetric
        from tigas.models.tigas_model import TIGASModel
        model = TIGASModel(img_size=256).eval()
        metric = TIGASMetric(model=model, device='cpu')
        x = torch.randn(2, 3, 256, 256)
        scores = metric(x)
        assert scores.shape == (2, 1)
    test("TIGASMetric forward", test_tigas_metric)
    
    # ==================== 6. Utils ====================
    print("\n[6/6] Testing utils...")
    
    def test_input_processor():
        from tigas.utils.input_processor import InputProcessor
        processor = InputProcessor(img_size=256, device='cpu')
        x = torch.randn(1, 3, 256, 256)
        out = processor.process(x)
        assert out.shape == (1, 3, 256, 256)
    test("InputProcessor", test_input_processor)
    
    def test_config():
        from tigas.utils.config import get_default_config
        config = get_default_config()
        assert 'model' in config
        assert 'training' in config
    test("Config utils", test_config)
    
    # ==================== Summary ====================
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed ({elapsed:.1f}s)")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    print("\n✓ All quick tests passed!")


if __name__ == "__main__":
    main()
