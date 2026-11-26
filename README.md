# MSGF_Test — MSGF-TIR Thermal Infrared Image Enhancement (Qt 6 / C++17)

Ứng dụng C++17 / Qt 6 và công cụ CLI cài đặt, trực quan hóa và kiểm định thuật toán **Tăng cường ảnh hồng ngoại nhiệt dựa trên lọc dẫn hướng đa tỉ lệ (MSGF-TIR)** theo đúng bài báo khoa học:

> **"Thermal Infrared-Image-Enhancement Algorithm Based on Multi-Scale Guided Filtering"**  
> *Huaizhou Li, Shuaijun Wang, Sen Li, Hong Wang, Shupei Wen, and Fengyu Li*  
> *Fire* **2024**, *7*(6), 192 — [https://doi.org/10.3390/fire7060192](https://doi.org/10.3390/fire7060192)

---

## 1. Bảng kiểm kê thuật toán & số liệu từ bài báo

| Thành phần | Trạng thái trong bài báo | Chi tiết cài đặt trong `MSGF_Test` |
|---|---|---|
| **Cấu trúc phân rã 3 tầng** | Có 100% (Section 2.1, Fig. 1–2) | Lọc dẫn hướng tự dẫn (Self-Guided Filter) nối tiếp 3 tầng tạo $B_1, B_2, B_3$ và $D_1, D_2, D_3$ trong [MsgfAlgorithm.cpp](core/MsgfAlgorithm.cpp). |
| **Công thức Self-Guided Filter** | Có 100% (Eq. 7–9) | Hệ số $a_k, b_k$ tính bằng `cv::boxFilter` (`BORDER_REFLECT`) trên miền chuẩn hóa $[0, 1]$, không phụ thuộc `opencv_contrib`. |
| **Xử lý Base Layer ($B_3$)** | Có 100% (Section 2.2, Eq. 10–15) | CLAHE (`cv::createCLAHE`) chia lưới $n \times n$, cắt ngưỡng $T = c \cdot \frac{n_x n_y}{K}$, phân phối lại và nội suy song tuyến. |
| **Xử lý Detail Layers ($D_1, D_2, D_3$)** | Có 100% (Section 2.3, Eq. 16–17) | Biến đổi tuyến tính động quanh trung bình $\mu_j$ với hệ số tương phản $\alpha_j$ trên ma trận `CV_32FC1` có dấu. |
| **Hợp nhất ảnh (Weighted Fusion)** | Có 100% (Section 2.4, Eq. 18) | Gộp chi tiết $I_D = \sum_{j=1}^3 w_j D_{j,\text{enh}}$ và trộn tuyến tính $I_{\text{out}} = p \cdot I_B + q \cdot I_D$. |
| **Độ đo & Bảng mốc (Benchmark)** | Có 100% (Section 3, Eq. 19–21 & Table 1) | Đầy đủ **PSNR**, **IE**, **AG**, **PIQE** (Venkatanath 2015) trong [MsgfMetrics.cpp](core/MsgfMetrics.cpp) + bảng Table 1 gắn sẵn trên Form. |
| **Tham số cụ thể ($r_k, \varepsilon_k, n, c, \alpha_k, p, q$)** | Không công bố giá trị số cố định | Bài báo nêu nguyên lý chọn theo bối cảnh; toàn bộ 14 tham số được mở qua `QSpinBox`/`QDoubleSpinBox` trên Form và CLI flags. |

---

## 2. Kiến trúc pipeline MSGF-TIR (4 bước cốt lõi)

```text
                    [Ảnh hồng ngoại đầu vào I (8-bit / 16-bit -> CV_32F [0,1])]
                                                │
          ┌────────────────────────────┬────────┴────────┐
          ▼                            │                 │
  [GF(I, r1, eps1)] ──► B1             │(I - B1)         │
          │                            ▼                 │
          │                        [D1: Small]           │
          ▼                            │                 │
  [GF(B1, r2, eps2)] ──► B2   (B1-B2)  │                 │
          │                 ────────► [D2: Medium]       │
          ▼                            │                 │
  [GF(B2, r3, eps3)] ──► B3   (B2-B3)  │                 │
          │                 ────────► [D3: Large]        │
          │                            │                 │
          ▼                            ▼                 │
  [CLAHE(c, n×n)]            [Dynamic Linear Eq.17]      │
      (Eq. 10-15)               (α1, α2, α3, Eq.16-17)   │
          │                            │                 │
          ▼                            ▼                 │
       [I_B]                 [I_D = w1·D1' + w2·D2' + w3·D3']
          │                            │                 │
          └─────────────┬──────────────┘                 │
                        ▼
           [Weighted Linear Fusion (Eq. 18)]
                I_out = p · I_B + q · I_D
                        │
                        ▼
            [Clamp [0, 1] -> CV_8UC1 [0, 255]]
```

> Xem chi tiết đầy đủ chứng minh toán học của từng phương trình (Eq. 1–21) tại [docs/ALGORITHM_SPEC.md](docs/ALGORITHM_SPEC.md).

---

## 3. Tính năng của ứng dụng

* **Kiến trúc Module Khép kín & Nhúng được (`MsgfForm` kế thừa `QWidget`)**:
  * Chạy độc lập qua `MsgfTirApp` (`main.cpp`).
  * Hoặc liên kết tĩnh target `msgf_widget` để nhúng trực tiếp `new MsgfForm(parent)` vào bất kỳ `QTabWidget` / `QMainWindow` nào khác.
  * Tách riêng hoàn toàn thư viện lõi `msgf_core` (thuần C++17 + OpenCV, **không phụ thuộc Qt**) dùng chung cho cả GUI, CLI và Unit Tests.
* **Xử lý nền không treo UI**: Dùng `QtConcurrent::run` + `QFutureWatcher` kết hợp cơ chế `generation ID` và debounce `60 ms` — kéo `QSpinBox`/`QDoubleSpinBox` liên tục không làm đơ giao diện và tự loại bỏ khung cũ lỗi thời.
* **3 Tab trực quan hóa với Zoom / Pan đồng bộ (`ImageCanvas`)**:
  1. **Compare (Input vs Output)**: So sánh song song ảnh gốc (`input8U`) và ảnh tăng cường (`output8U`).
  2. **Decomposition (Figure 3)**: Lưới $2 \times 3$ hiển thị đồng thời `Original (I)`, `D1` (nhỏ), `D2` (vừa), `D3` (lớn) đã qua chuẩn hóa hiển thị `normalizeForDisplay`, lớp nền `B3` và lớp nền sau CLAHE `IB`.
  3. **Fused Details & Paper Ref**: Quan sát tổng chi tiết `ID_fused` và ảnh kết quả tham chiếu từ bài báo (`*_paper_ours.png`, nếu có).
  * *Tương tác chuột*: Lăn chuột giữa để zoom tại tâm trỏ chuột, kéo chuột trái để pan, **nhấp đúp chuột trái** để reset fit-to-window trên toàn bộ các khung ảnh.
* **Hỗ trợ đa định dạng & điều hướng nhanh**: Hỗ trợ 8-bit, 16-bit (`CV_16U`), BGR/BGRA, ảnh float (`CV_32F`/`CV_64F`); nạp file đơn lẻ, mở cả thư mục, điều hướng nhanh bằng nút `◀` / `▶` hoặc phím mũi tên **Left / Right**, xuất ảnh đơn (`Save Output...`) hoặc toàn bộ 10 tầng trung gian (`Export All Layers...`).
* **Đánh giá định lượng thời gian thực & đối chiếu Table 1**: Hiển thị **PSNR**, **IE** (`input -> output`), **AG** (`input -> output`), **PIQE** (`input -> output`, chuẩn Venkatanath 2015), Runtime (`ms`) và bảng đối chiếu **Table 1** (12 dòng: Scene 1–3 × HE, GF&DDE, BF&DDE, MSGF-TIR Ours).

---

## 4. Cấu trúc thư mục dự án

```text
MSGF_Test/
├── CMakeLists.txt              # Cấu hình build C++17: msgf_core, msgf_widget, MsgfTirApp, msgf_cli, msgf_tests
├── README.md                   # Tài liệu tổng quan, hướng dẫn build/chạy & bảng tham số
├── LICENSE                     # Giấy phép MIT
├── .gitignore
├── run_gui.sh                  # Script tự động cấu hình, biên dịch Release và khởi chạy MsgfTirApp
├── main.cpp                    # Entry point độc lập bọc MsgfForm trong QMainWindow
├── core/
│   ├── MsgfParams.h            # Cấu trúc MsgfParams, MsgfOutputs và dữ liệu tham chiếu Table 1
│   ├── MsgfAlgorithm.h/.cpp    # Thuần C++17 & OpenCV: 3-scale Self-GF, CLAHE, Dynamic Linear, Fusion
│   └── MsgfMetrics.h/.cpp      # Đo PSNR, IE, AG (Eq. 19–21) và PIQE (Venkatanath 2015)
├── ui/
│   ├── MsgfForm.ui             # File giao diện Qt Designer (QSplitter 3 cột, QScrollArea, QTabWidget, Table 1)
│   ├── MsgfForm.h/.cpp         # Signal/Slot, background QtConcurrent worker, export & đồng bộ view
│   └── ImageCanvas.h/.cpp      # Custom QWidget hiển thị ảnh zoom/pan/double-click reset
├── cli/
│   └── msgf_cli.cpp            # Công cụ dòng lệnh (CLI) benchmark batch & xuất CSV / từng tầng trung gian
├── tests/
│   └── test_msgf.cpp           # Unit test kiểm tra bất biến toán học (CTest)
├── docs/
│   └── ALGORITHM_SPEC.md       # Tài liệu đặc tả toán học chi tiết Eq. (1)–(21)
├── sample_images/              # 3 cảnh mẫu & Figure 3 tách sạch từ PDF bài báo (Scene 1–3)
└── results/                    # Thư mục lưu kết quả test (tùy chọn)
```

---

## 5. Hướng dẫn cài đặt, biên dịch & chạy

### Yêu cầu hệ thống
* Trình biên dịch hỗ trợ **C++17** (Clang, GCC hoặc MSVC).
* **CMake** $\ge 3.20$.
* **OpenCV** $\ge 4.x$ (hoặc $5.x$) với các module `core`, `imgproc`, `imgcodecs` (không cần `opencv_contrib`).
* **Qt 6** (`Core`, `Gui`, `Widgets`, `Concurrent`) — có thể tắt bằng `-DMSGF_BUILD_GUI=OFF` nếu chỉ cần build core + CLI.

### Biên dịch và chạy nhanh trên macOS (Homebrew Qt 6 & OpenCV)

```bash
# Cách 1: Chạy script tự động build + mở GUI
./run_gui.sh

# Cách 2: Build thủ công + chạy unit tests + CLI
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$(brew --prefix qt);$(brew --prefix opencv)"
cmake --build build --config Release -j"$(sysctl -n hw.ncpu)"

# Chạy toàn bộ unit tests (CTest)
ctest --test-dir build --output-on-failure

# Chạy GUI (tùy chọn truyền đường dẫn ảnh khởi đầu)
./build/MsgfTirApp
```

### Sử dụng CLI Benchmark (`msgf_cli`)

```bash
# Benchmark mặc định trên thư mục sample_images/
./build/msgf_cli --input sample_images

# Chạy trên thư mục dataset FLIR/LLVIP, xuất ảnh tăng cường, các tầng phân rã và báo cáo CSV
./build/msgf_cli --input /path/to/thermal_dir \
  --output-dir results/flir_out --save-layers \
  --csv results/flir_metrics.csv \
  --r1 2 --eps1 0.005 --r2 6 --eps2 0.02 --r3 16 --eps3 0.08 \
  --clip 2.0 --grid 8 --alpha1 1.5 --alpha2 1.5 --alpha3 1.5 \
  --p 1.0 --q 1.0
```

---

## 6. Kết quả đối chiếu & kiểm thử thực nghiệm

### A. Số liệu tham chiếu từ bài báo (Table 1 — *Fire* 2024, 7, 192)

| Cảnh kiểm thử (Dataset) | Thuật toán | PSNR (dB) ↑ | IE ↑ | AG ↑ | PIQE ↓ |
|---|---|---:|---:|---:|---:|
| **Scene 1** *(iRay T3-Pro)* | HE | 13.75 | 6.95 | 86.20 | 38.79 |
| | GF&DDE | 23.42 | 6.93 | 48.72 | 38.91 |
| | BF&DDE | 27.07 | 7.05 | 49.22 | 66.21 |
| | **MSGF-TIR (Ours)** | **27.07** | **7.34** | **54.15** | **34.45** |
| **Scene 2** *(FLIR)* | HE | 12.33 | 5.91 | 61.83 | 48.54 |
| | GF&DDE | 22.83 | 6.09 | 18.43 | 11.32 |
| | BF&DDE | 19.56 | 6.11 | 19.48 | 42.97 |
| | **MSGF-TIR (Ours)** | **21.20** | **6.47** | **19.84** | **9.35** |
| **Scene 3** *(FLIR)* | HE | 11.80 | 5.66 | 86.92 | 46.18 |
| | GF&DDE | 18.57 | 5.70 | 15.88 | 9.38 |
| | BF&DDE | 21.46 | 5.79 | 15.79 | 47.56 |
| | **MSGF-TIR (Ours)** | **30.74** | **6.16** | **19.05** | **8.02** |

### B. Kết quả chạy thực tế `msgf_cli` với cấu hình mặc định trên `sample_images/` (ảnh $275 \times 207$ trích từ PDF)

| File ảnh mẫu | Runtime (ms) | PSNR (dB) | IE (`in → out`) | AG (`in → out`) | PIQE (`in → out`) |
|---|---:|---:|---:|---:|---:|
| `scene1_original.png` | 2.74 | 21.80 | 7.04 → **7.48** | 6.37 → **9.70** | 15.16 → **10.30** |
| `scene2_original.png` | 1.65 | 27.70 | 5.23 → **6.66** | 3.09 → **4.75** | 24.67 → **22.83** |
| `scene3_original.png` | 1.82 | 25.58 | 4.95 → **6.44** | 3.41 → **5.31** | 19.41 → **15.33** |
| **Trung bình (3 ảnh)** | **2.07** | **25.03** | **6.86** | **6.59** | **16.15** |

> **Lưu ý về sai khác trị số tuyệt đối so với Table 1**: Các file trong `sample_images/` được trích trực tiếp từ PDF bài báo ở kích thước hình minh họa ($275 \times 207$ có nén JPEG của PDF), trong khi số liệu Table 1 của nhóm tác giả được đo trên ảnh cảm biến gốc độ phân giải đầy đủ (iRay T3-Pro và FLIR gốc). Tuy nhiên xu hướng cải thiện trên cả 3 cảnh đều nhất quán: **IE tăng**, **AG tăng rõ rệt**, và **PIQE giảm** (chất lượng cảm nhận tốt hơn).

---

## 7. Hướng dẫn tinh chỉnh tham số & Lưu ý kỹ thuật

| Tham số | Vai trò | Mặc định | Tác động khi điều chỉnh |
|---|---|---:|---|
| $r_1, r_2, r_3$ | Bán kính cửa sổ lọc GF 3 tầng | `2, 6, 16` | Tăng dần từ nhỏ $\rightarrow$ vừa $\rightarrow$ lớn để tách chi tiết mịn, trung bình và cấu trúc thô. |
| $\varepsilon_1, \varepsilon_2, \varepsilon_3$ | Hệ số chính quy hóa trên miền $[0, 1]$ | `0.005, 0.02, 0.08` | $\varepsilon = 0$ trả về ảnh gốc; $\varepsilon$ càng lớn vùng phẳng càng mịn, chi tiết giữ lại trong $D_j$ càng nhiều. |
| `clipLimit` ($c$) | Ngưỡng cắt histogram CLAHE | `2.0` | Tăng $c$ tăng tương phản cục bộ lớp nền $B_3$, nhưng quá cao có thể làm nổi nhiễu vùng trời/tối. |
| `tileGrid` ($n$) | Lưới chia khối CLAHE ($n \times n$) | `8` | Chia nhỏ hơn ($n$ lớn hơn) thích nghi cục bộ mạnh hơn; $n$ nhỏ tiến gần cân bằng toàn cục. |
| $\alpha_1, \alpha_2, \alpha_3$ | Hệ số Dynamic Linear Enhancement | `1.5, 1.5, 1.5` | $\alpha_j > 1$ khuếch đại tương phản chi tiết tầng $j$; $\alpha_j < 1$ làm dịu chi tiết/nhiễu tần số cao. |
| $w_1, w_2, w_3$ | Trọng số gộp 3 tầng chi tiết | `1.0, 1.0, 1.0` | Điều chỉnh tỷ trọng giữa biên sắc nét ($D_1$) và khối cấu trúc trung/lớn ($D_2, D_3$). |
| $p, q$ | Hệ số trộn $I_{\text{out}} = p I_B + q I_D$ | `1.0, 1.0` | $p$ kiểm soát độ sáng nền CLAHE; $q$ kiểm soát cường độ chi tiết tổng hợp đưa vào ảnh cuối. |

### Ba lưu ý kỹ thuật quan trọng:
1. **Đơn vị của $\varepsilon$ gắn với miền cường độ ảnh**: `MsgfAlgorithm` chuẩn hóa ảnh về miền $[0, 1]$ (`CV_32F` chia $255$), nên $\varepsilon \in [10^{-4}, 10^{-1}]$. Nếu tính trên miền $[0, 255]$, phương sai $\sigma_k^2$ tăng gấp $255^2 = 65025$ lần nên $\varepsilon$ phải nhân tương ứng $65025$.
2. **Giữ kiểu float có dấu (`CV_32FC1`) cho $D_1, D_2, D_3$**: Vì $D_j = B_{j-1} - B_j$ mang cả giá trị âm (viền tối) lẫn dương (viền sáng). Tuyệt đối không ép về `CV_8U` trước bước hợp nhất; chỉ dùng `MsgfAlgorithm::normalizeForDisplay()` khi vẽ lên UI.
3. **Cách nhúng `MsgfForm` vào ứng dụng Qt khác**:

```cmake
# Trong CMakeLists.txt của project chính:
add_subdirectory(path/to/MSGF_Test)
target_link_libraries(YourMainApp PRIVATE msgf_widget)
```

```cpp
// Trong MainWindow hoặc QTabWidget của bạn:
#include "ui/MsgfForm.h"

auto* msgfTab = new MsgfForm(tabWidget);
tabWidget->addTab(msgfTab, tr("MSGF-TIR Enhancement"));
// Có thể truyền ảnh trực tiếp từ luồng xử lý hiện tại:
msgfTab->setInputImage(thermalMat8Uor16U, "Thermal Input");
```

---

## 8. Giấy phép (License)

Phát hành theo giấy phép **MIT License** — xem chi tiết tại [LICENSE](LICENSE). Phục vụ nghiên cứu khoa học, kiểm chứng thuật toán và phát triển hệ thống xử lý ảnh hồng ngoại nhiệt.
