#include <torch/extension.h>
#include <algorithm>

// Функція приймає два тензори PyTorch (списки рамок) і повертає тензор з результатами
torch::Tensor calculate_iou(torch::Tensor boxes_a, torch::Tensor boxes_b) {
    
    // Отримуємо розміри тензорів (кількість рамок у кожному списку)
    int num_a = boxes_a.size(0);
    int num_b = boxes_b.size(0);

    // Створюємо порожній тензор для результатів, заповнений нулями, розміром [num_a, num_b]
    auto result = torch::zeros({num_a, num_b});

    // Створюємо "аксесори" (accessors). 
    // Це фішка PyTorch C++ API, яка дає сирому C++ надшвидкий прямий доступ до пам'яті тензора.
    // float - тип даних, 2 - кількість вимірів (матриця).
    auto a = boxes_a.accessor<float, 2>();
    auto b = boxes_b.accessor<float, 2>();
    auto res = result.accessor<float, 2>();

    // Погнали в жорсткі C++ цикли, які виконаються за мілісекунди
    for (int i = 0; i < num_a; ++i) {
        for (int j = 0; j < num_b; ++j) {
            
            // 1. Шукаємо координати перетину (за нашою формулою)
            float x1 = std::max(a[i][0], b[j][0]);
            float y1 = std::max(a[i][1], b[j][1]);
            float x2 = std::min(a[i][2], b[j][2]);
            float y2 = std::min(a[i][3], b[j][3]);

            // 2. Рахуємо площу перетину. std::max не дає площі стати від'ємною, якщо рами не перетинаються.
            float inter_w = std::max(0.0f, x2 - x1);
            float inter_h = std::max(0.0f, y2 - y1);
            float inter_area = inter_w * inter_h;

            // 3. Рахуємо площі оригінальних рамок
            float area_a = (a[i][2] - a[i][0]) * (a[i][3] - a[i][1]);
            float area_b = (b[j][2] - b[j][0]) * (b[j][3] - b[j][1]);

            // 4. Записуємо результат за формулою IoU
            if (inter_area > 0) {
                res[i][j] = inter_area / (area_a + area_b - inter_area);
            }
        }
    }
    
    return result;
}

// МАГІЯ ТУТ: Цей макрос "обгортає" наш C++ код, щоб Python міг його побачити
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("calculate_iou", &calculate_iou, "Надшвидкий розрахунок IoU на чистому C++");
}