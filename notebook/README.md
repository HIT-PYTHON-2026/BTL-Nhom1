# Bài toán phân loại cảm xúc - Emotion classification

## Giới thiệu bài toán
Phân loại cảm xúc qua hình ảnh là quá trình sử dụng các thuật toán máy tính để nhận diện biểu cảm khuôn mặt và gán chúng vào các trạng thái tâm lý tương ứng. Đây là bài toán kết hợp giữa xử lý ảnh và học sâu (Deep Learning).

## Mục tiêu Bài toán

Mục tiêu của dự án là phát triển một hệ thống có khả năng tự động trích xuất đặc trưng khuôn mặt và phân loại cảm xúc từ hình ảnh/video đầu vào. Bên cạnh đó, dự án còn là cơ hội để nhóm sinh viên nghiên cứu và làm chủ các kiến thức chuyên sâu về Deep Learning, xử lý ảnh, quy trình phát triển phần mềm và tư duy giải quyết vấn đề.

## Nhóm thực hiện: Nhóm 1

| STT  | Họ tên           | Facebook                                                 |
| :--- | :--------------- | :------------------------------------------------------- |
| 1    | Trần Văn Sơn     | [Sơn Trần](https://www.facebook.com/son.tran.384324)     |
| 2    | Hoàng Thanh Diệu | [Diệu Hoàng](https://www.facebook.com/dieu.hoang.12457/) |
| 3    | Vũ Trang Ngân    | [Trang Ngân](https://www.facebook.com/nony.mous.77736)   |

## Cấu trúc thư mục
![Project Structure](/image.png)

## Các chức năng chính

### Chức năng chính cho người dùng
> * Tải ảnh/Sử dụng Camera
> * Phân tích cảm xúc tức thời
> * Lịch sử phân tích
> * Gợi ý dựa trên cảm xúc
> * Chia sẻ kết quả 

### Chức năng quản trị viên (Nếu có)
> * Quản lý bộ dữ liệu
> * Giám sát hiệu năng mô hình
> * Quản lý phiên bản mô hình
> * Thống kê hệ thống
> * Phê duyệt phản hồi

## Demo sản phẩm

-  Người dùng:
----------------
> * Tải ảnh

![](/image.png)
 

 -  Quản trị viên:
----------------
> * Tải ảnh

![](/image.png)

 ## Các ngôn ngữ, công nghệ sử dụng
 > * Ngôn ngữ sử dụng: `Python`
 > * IDE sử dụng: `Visual Studio Code`
 > * Thư viện Deep Learning: `PyTorch`, `NumPy`, `Matplotlib`
 > * Thư viện xử lý hình ảnh: `Torchvision Transforms`
 > * Công cụ lập trình giao diện: `Streamlit`
 > * Công cụ quản lý môi trường và thư viện: `Conda`
 > * Công cụ quản lý phiên bản: `Git`
 > * Công cụ quản lý mã nguồn: `Github`

## Hướng dẫn cài đặt chương trình
> * **Bước 1:** Clone project [Emotion_Classification](https://github.com/HIT-PYTHON-2026/BTL-Nhom1.git)
> * **Bước 2:** Tạo và kích hoạt môi trường ảo `python -m venv venv`
> * **Bước 3:** Cài đặt các thư viện cần thiết từ file cấu hình: `pip install -r requirements.txt`
> * **Bước 4:** Sau đó di chuyển vào thư mục dự án: `cd frontend`
> * **Bước 5:** Bước 5: Khởi chạy ứng dụng: `python app.py`
> * ***Note:*** Xem file hướng dẫn cài đặt để hiểu rõ thêm: [File hướng dẫn chi tiết](https://docs.google.com/document/d/1o6tw7wAYEVP2A2WoEZ1EYq1Wg9a530sO8681uQ7o4z4/edit?usp=sharing)

## Tài liệu tham khảo

- [Các phương pháp đánh giá một hệ thống phân lớp](https://machinelearningcoban.com/2017/08/31/evaluation/)