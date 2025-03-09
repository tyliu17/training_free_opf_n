# NAS FOR training_free
from training_free import generate_architectures, train_and_evaluate, save_benchmark, train_surrogate_model, predict_performance



# 生成架構
architectures = generate_architectures(2600)



# 訓練 80% 的架構，並儲存 Benchmark
results = train_and_evaluate(architectures[:2080], input_size=10, output_size=1)
print(results)
print(dfsdfsdfdsfsdfd)
save_benchmark(results)

# 訓練代理模型
train_surrogate_model()

# 預測新架構的效能（Training-Free NAS）
new_arch = architectures[2081]
predicted_MSE = predict_performance(new_arch)
print(f"預測 MSE: {predicted_MSE}")