# Đặc tả Toán học & Cài đặt MSGF-TIR (Fire 2024, 7, 192)

Tài liệu này ánh xạ chi tiết từng công thức từ bài báo **Li et al., *"Thermal Infrared-Image-Enhancement Algorithm Based on Multi-Scale Guided Filtering"*, Fire 2024, 7, 192** sang mã nguồn C++17 / OpenCV trong thư mục `core/`.

---

## 1. Phân rã đa tỉ lệ bằng Self-Guided Filter (Section 2.1, Eq. 1–9)

### 1.1 Mô hình tuyến tính cục bộ

Với ảnh đầu vào $I$ được chuẩn hóa về miền $[0, 1]$ (`CV_32FC1`), bộ lọc dẫn hướng tự dẫn (Self-Guided Filter) coi ảnh đầu vào chính là ảnh dẫn hướng ($p \equiv I$). Trong cửa sổ vuông $\omega_k$ tâm $k$ bán kính $r$ (kích thước $(2r+1) \times (2r+1)$), đầu ra $q$ liên hệ tuyến tính với $I$:

$$q_i = a_k I_i + b_k, \quad \forall i \in \omega_k \tag{2}$$

Hàm chi phí cực tiểu hóa sai số tái tạo kèm thành phần chính quy $\varepsilon$:

$$E(a_k, b_k) = \sum_{i \in \omega_k} \left((a_k I_i + b_k - I_i)^2 + \varepsilon a_k^2\right) \tag{4}$$

Khi ảnh dẫn hướng là chính ảnh đầu vào, nghiệm bình phương tối thiểu rút gọn thành:

$$a_k = \frac{\sigma_k^2}{\sigma_k^2 + \varepsilon}, \quad b_k = (1 - a_k)\mu_k \tag{8–9}$$

Trong đó $\mu_k$ và $\sigma_k^2 = \mathbb{E}[I^2]_{\omega_k} - \mu_k^2$ lần lượt là kỳ vọng và phương sai cục bộ trong $\omega_k$ (kẹp $\sigma_k^2 \ge 0$ để triệt tiêu sai số dấu phẩy động ở vùng đồng nhất). Do mỗi điểm ảnh $i$ thuộc nhiều cửa sổ $\omega_k$, giá trị lọc cuối cùng lấy trung bình qua mọi cửa sổ chứa nó:

$$q_i = \bar{a}_i I_i + \bar{b}_i \tag{7}$$

**Trường hợp biên $\varepsilon = 0$ hoặc $r = 0$**: Theo Section 2.1, khi $\varepsilon = 0$, $q_i = I_i$ (không lọc). Trong `MsgfAlgorithm::selfGuidedFilter`, điều kiện `if (radius <= 0 || eps <= 0.0) return src32F.clone();` đảm bảo không phát sinh phép chia `0.0f / 0.0f` (`NaN`) tại các vùng phẳng có $\sigma_k^2 = 0$.

### 1.2 Phân rã nối tiếp 3 tầng (Fig. 2)

$$\begin{aligned}
B_1 &= \text{GF}(I, r_1, \varepsilon_1), & D_1 &= I - B_1 \\
B_2 &= \text{GF}(B_1, r_2, \varepsilon_2), & D_2 &= B_1 - B_2 \\
B_3 &= \text{GF}(B_2, r_3, \varepsilon_3), & D_3 &= B_2 - B_3
\end{aligned}$$

Tính chất bảo toàn đại số trước bước tăng cường:

$$B_3 + D_1 + D_2 + D_3 = I$$

Bất biến này được kiểm chứng tự động trong [tests/test_msgf.cpp](../tests/test_msgf.cpp) với ngưỡng sai số `< 1e-5`.

---

## 2. Tăng cường lớp nền $B_3$ bằng CLAHE (Section 2.2, Eq. 10–15)

Lớp nền $B_3$ chứa phân bố độ sáng dải rộng và vùng tối nền. Thuật toán thực hiện:
1. Kẹp $B_3$ về $[0, 1]$ và lượng tử hóa về 8-bit (`CV_8UC1`).
2. Chia lưới $n \times n$ khối không chồng lấn (`claheTileGrid`).
3. Tính ngưỡng cắt histogram theo Eq. (10):
   $$T = c \times \frac{n_x n_y}{K}$$
   với $c$ là `claheClipLimit`, $n_x \times n_y$ là số điểm ảnh mỗi khối, $K = 256$ mức xám.
4. Tính tổng phần vượt ngưỡng $S = \sum_{x=0}^{K-1} \max(h(x) - T, 0)$ và phân bổ đều $A = S / K$ cho toàn bộ các bin theo Eq. (11)–(13).
5. Cân bằng cục bộ kết hợp nội suy tuyến tính tại 4 cạnh và nội suy song tuyến tại vùng trung tâm (Eq. 14–15) qua `cv::createCLAHE(clipLimit, Size(n, n))`, sau đó đưa về `CV_32FC1` tạo $I_B \in [0, 1]$.

---

## 3. Tăng cường các lớp chi tiết $D_1, D_2, D_3$ (Section 2.3, Eq. 16–17)

Mỗi lớp chi tiết có dấu $D_j$ ($j \in \{1, 2, 3\}$) được tăng cường bằng **Dynamic Linear Enhancement**:

$$\mu_j = \frac{1}{N} \sum_{i=1}^{N} D_j(i) \tag{16}$$

$$D_{j,\text{enh}}(x, y) = \alpha_j \cdot \bigl(D_j(x, y) - \mu_j\bigr) + \mu_j \tag{17}$$

Nhận xét toán học:
* Phép biến đổi giữ nguyên kỳ vọng $\mathbb{E}[D_{j,\text{enh}}] = \mu_j$ nên **không làm trôi độ sáng nền** (brightness shift).
* Độ lệch chuẩn tăng tỷ lệ đúng bằng $|\alpha_j|$: khi $\alpha_j > 1$, độ tương phản biên/kết cấu tăng đều quanh tâm phân bố $\mu_j$; khi $\alpha_j = 1$, tầng chi tiết được giữ nguyên.

---

## 4. Hợp nhất ảnh tuyến tính có trọng số (Section 2.4, Eq. 18)

Ba tầng chi tiết sau tăng cường được gộp theo trọng số tầng $(w_1, w_2, w_3)$ và trộn với lớp nền $I_B$:

$$I_D = w_1 D_{1,\text{enh}} + w_2 D_{2,\text{enh}} + w_3 D_{3,\text{enh}}$$

$$I_{\text{out}} = \text{clamp}_{[0, 1]}\bigl(p \cdot I_B + q \cdot I_D\bigr) \tag{18}$$

Cuối cùng $I_{\text{out}}$ được quy đổi về `CV_8UC1` trên dải $[0, 255]$.

---

## 5. Các độ đo đánh giá chất lượng (Section 3, Eq. 19–21 & PIQE)

Cài đặt trong [core/MsgfMetrics.cpp](../core/MsgfMetrics.cpp):

1. **PSNR — Peak Signal-to-Noise Ratio (Eq. 19)**:
   $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{255^2 \cdot M \cdot N}{\sum_{i=0}^{M-1}\sum_{j=0}^{N-1}(I(i,j) - I_m(i,j))^2}\right)$$
2. **IE — Information Entropy (Eq. 20)**:
   $$H(x) = -\sum_{i=0}^{255} p(x_i)\log_2 p(x_i)$$
3. **AG — Average Gradient (Eq. 21)**:
   $$G_{\text{avg}} = \frac{1}{(M-1)(N-1)} \sum_{i=0}^{M-2}\sum_{j=0}^{N-2} \sqrt{\frac{(H(i+1,j) - H(i,j))^2 + (H(i,j+1) - H(i,j))^2}{2}}$$
4. **PIQE — Perception-based Image Quality Evaluator (Venkatanath et al., NCC 2015)**:
   Tính hệ số MSCN (Gaussian $7 \times 7$, $\sigma = 7/6$), phân tích các khối $16 \times 16$ có hoạt động cục bộ ($\sigma_{\text{blk}}^2 > 0.1$) để đánh giá độ biến dạng cấu trúc / chặn khối và nhiễu không tương quan. Điểm nằm trong $(0, 100]$, **càng thấp chất lượng cảm nhận càng cao**.
