import torch
import time
import fast_cv_ops  # Наш власний C++ модуль!

# Функція для генерації випадкових рамок у форматі [x1, y1, x2, y2]
def generate_boxes(num_boxes):
    boxes = torch.rand((num_boxes, 4)) * 100
    # Робимо так, щоб права нижня координата завжди була більшою за ліву верхню
    boxes[:, 2] += boxes[:, 0] 
    boxes[:, 3] += boxes[:, 1]
    return boxes

# 1. Створюємо тестові дані
num_a = 2500
num_b = 2500
print(f"Генеруємо {num_a} рамок A та {num_b} рамок B (всього {num_a * num_b} порівнянь)...")
boxes_a = generate_boxes(num_a)
boxes_b = generate_boxes(num_b)

# 2. Чистий Python (наївна реалізація в циклах, один в один як у нашому C++ коді)
def python_iou(boxes_a, boxes_b):
    num_a = boxes_a.size(0)
    num_b = boxes_b.size(0)
    result = torch.zeros((num_a, num_b))
    
    for i in range(num_a):
        for j in range(num_b):
            x1 = max(boxes_a[i][0], boxes_b[j][0])
            y1 = max(boxes_a[i][1], boxes_b[j][1])
            x2 = min(boxes_a[i][2], boxes_b[j][2])
            y2 = min(boxes_a[i][3], boxes_b[j][3])
            
            inter_w = max(0.0, x2 - x1)
            inter_h = max(0.0, y2 - y1)
            inter_area = inter_w * inter_h
            
            area_a = (boxes_a[i][2] - boxes_a[i][0]) * (boxes_a[i][3] - boxes_a[i][1])
            area_b = (boxes_b[j][2] - boxes_b[j][0]) * (boxes_b[j][3] - boxes_b[j][1])
            
            if inter_area > 0:
                result[i][j] = inter_area / (area_a + area_b - inter_area)
    return result

# Розігрів (щоб PyTorch ініціалізував усі внутрішні процеси і тест був чесним)
_ = fast_cv_ops.calculate_iou(boxes_a[:10], boxes_b[:10])

# --- ТЕСТ PYTHON ---
print("\nЗапуск чистого Python...")
start_py = time.time()
res_py = python_iou(boxes_a, boxes_b)
time_py = time.time() - start_py
print(f"Час Python: {time_py:.4f} секунд")

# --- ТЕСТ C++ ---
print("\nЗапуск C++ розширення...")
start_cpp = time.time()
res_cpp = fast_cv_ops.calculate_iou(boxes_a, boxes_b)
time_cpp = time.time() - start_cpp
print(f"Час C++: {time_cpp:.4f} секунд")

# 3. Перевірка на чесність
# Рахуємо максимальну різницю між результатами двох функцій (має бути 0)
diff = torch.max(torch.abs(res_py - res_cpp))
print(f"\nМаксимальна похибка між Python та C++: {diff:.6f}")

if time_cpp > 0:
    print(f"🔥 C++ швидший у {time_py / time_cpp:.1f} разів!")