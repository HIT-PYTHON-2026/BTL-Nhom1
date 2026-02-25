# Tiền xử lý và tăng cường dữ liệu (Data Augmentation)
Data train:
+ Random Resized Crop: Cắt ngẫu nhiên một phần ảnh và resize về kích thước tiêu chuẩn (224x224). Giúp mô hình nhận diện được các đặc điểm khuôn mặt ngay cả khi không nằm chính giữa khung hình.
+ Random Horizontal Flip: Lật ảnh ngang (tỉ lệ 50%) để xử lý sự đối xứng của khuôn mặt.
+ Color Jitter (p=0.8): Điều chỉnh ngẫu nhiên độ sáng, độ tương phản, độ bão hòa và sắc thái. Điều này giúp mô hình chống chọi với các điều kiện ánh sáng khác nhau.
+ Random Grayscale (p=0.2): Chuyển ảnh sang đen trắng giúp mô hình tập trung vào hình thái (shapes) thay vì quá phụ thuộc vào màu sắc.
+ Random Rotation (20°): Xoay ảnh một góc tối đa 20 độ để mô phỏng các tư thế nghiêng đầu.
+ Normalization: Chuẩn hóa theo ImageNet mean và std:
	+ Mean: [0.485, 0.456, 0.406]
	+ Std: [0.229, 0.224, 0.225]
+ Random Erasing (p=0.25): Xóa ngẫu nhiên một vùng nhỏ trên ảnh. Kỹ thuật này buộc mô hình phải học từ các phần khác nhau của khuôn mặt (ví dụ: nhận diện qua mắt khi miệng bị che).

Data test:
+ Resize: Đưa về cùng kích thước (224x224) với data train.
+ Normalization: Sử dụng cùng bộ thông số chuẩn hóa với tập huấn luyện.
Cấu hình DataLoader
+ Batch Size: 32 (Có thể điều chỉnh tùy theo bộ nhớ GPU).
+ Num Workers: 4 (Sử dụng đa luồng để tăng tốc độ load dữ liệu).
+ Pin Memory: Được kích hoạt để tăng tốc độ truyền dữ liệu từ CPU lên GPU.

# Model Resnet18
Tổng quan (overview):
conv3x3: Là lớp tích chập được xây dựng để tái sử dụng lại nhiều lần
Block: Đơn vị cơ bản của ResNet, chứa các Skip Connection giúp truyền tín hiệu trực tiếp qua các lớp
Resnet: Là class khai báo và xây dựng cấu trúc của 1 mạng neutral network (ở đây là 1 mạng gần giống resnet18)
Resnet18: Là hàm khai báo mạng neutral network (có 2 tham số)
+ num_classes: Số lượng nhãn của model (7: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
+ parameter_dropout: Tham số truyền vào layer dropout ở output layer (classification head)
Các layer được sử dụng:
+ Convolution layer: Tìm cách đặc trưng trong ảnh
+ batch normalization: Vì output của layer này là input của layer tiếp theo nên mỗi khi weight của layer trước thay đổi dẫn đến output của nó cũng thay đổi (nó như phản ứng dây truyền) vậy nên bn1 xuất hiện với nhiệm vụ giúp model hội tụ nhanh hơn, ổn định hơn, chống overfitting
+ Activation layer (còn được gọi là Activations functions) (ReLU): Thay thế các đặc trưng xấu (bé hơn 0) bằng 0
+ Pooling layer: Có tác dụng giảm kích thước ảnh, giảm số lượng tham số cho model, ổn định hóa các đặc trưng (giữ lại các đặc trưng chính và giảm các chi tiết không cần thiết cho các layer tiếp theo)
	+ MaxPooling: Tìm các đặc trưng nổi bật (Trả về giá trị lớn nhất của 1 vùng kernel)
	+ AvgPooling: Giảm độ nhiễu của ảnh và làm mượt các đặc trưng bằng cách lấy giá trị trung bình, giúp tổng hợp thông tin toàn cục trước khi đưa vào lớp phân loại (Trả về giá trị 	trung bình cộng của 1 vùng kernel)
+ Dropout layer(p): Random (p * 100%) phần tử input thành 0 nhằm tránh hiện tượng model học thuộc data train
+ Fully connected layer: Thực hiện phép nhân ma trận để kết hợp các đặc trưng bậc cao đã học được thành các điểm số tương ứng cho từng nhãn

Cấu trúc chi tiết:
+ Block(padding = 1): 
input(x) -> conv1 -> bn1 -> relu -> conv2 -> bn2 -> output
final output: relu(Downsample(x) + output) (skip connection: Giúp giải quyết vấn đề triệt tiêu đạo hàm (vanishing gradient) trong các mạng có cấu trúc phức tạp)
+ Downsample: Làm cho ảnh input(x) có kích thước và số channels bằng output (Dowsample sẽ xuất hiện tại Block[0] tại layer["2","3","4"])(intput(x) -> conv1 -> bn1 -> )

Resnet: input(batch, 224,224,3)
+ The Stem (conv, bn, relu, maxpool) output: (batch,56,56,64)
  + conv(outputchannels(số lượng kernel) = 64,kernel_size = 7, stride = 2, padding = 3): Tìm các đặc trưng cơ bản như: góc, cạnh (output: (batch,112,112,64))
	+ bn(64): 
	+ relu:
	+ maxpooling: output: (batch, 56,56,64)
+ Residual: 
  + layer 1:
		+ Block[0](outchannels = 64, stride = 1): output: (batch,56,56,64)
		+ Block[1](outchannels = 64, stride = 1): output: (batch,56,56,64)
	+ layer 2:
	  + Block[0](outchannels = 128, stride = 2): output: (batch,28,28,128)
		+ Block[1](outchannels = 128, stride = 1): output: (batch,28,28,128)
	+ layer 3: 
	  + Block[0](outchannels = 256, stride = 2): output: (batch,14,14,256)
		+ Block[1](outchannels = 256, stride = 1): output: (batch,14,14,256)
	+ layer 4: 
	  + Block[0](outchannels = 512, stride = 2): output: (batch,7,7,512)
		+ Block[1](outchannels = 512, stride = 1): output: (batch,7,7,512)
+ AvgPooling(output(1,1)): output: (batch, 1,1,512)
+ Flatten : chuyển đổi tensor đa chiều (batch, 1, 1, 512) thành vector 1 chiều (batch, 1 * 1 * 512) để làm đầu vào cho lớp FC: output: (batch,512)
+ Classification head (dropout,Linear(Fully connected)):
  + dropout: Tắt ngẫu nhiên (p * 100%) trong số các đặc trưng
	+ linear: Tổng hợp các đặc trưng để đưa ra đầu ra là các Logits (điểm số thô), 7 giá trị Logits này sau đó sẽ được đưa qua hàm Softmax (thường tích hợp trong Loss function) để chuyển đổi thành xác suất dự báo cho 7 nhãn cảm xúc 

# Train loop

Quy trình hyperparameter_tuning: Sau bước tiền xử lý, hệ thống thực hiện tìm kiếm bộ tham số tối ưu thông qua framework Optuna kết hợp với kiến trúc ResNet18.
| Hyperparameter | Phạm vi | Kiểu dữ liệu |
| :--- | :--- | :--- |
| parameter_dropout | 0.3 - 0.5 | float |
| learning rate | 1e-6 - 1e-3 | float |
| label smoothing | 0.1 - 0.2 | float |
| weight decay |1e-6 - 1e-3 | float |
  + n_startup_trials (5): Chạy ít nhất 5 trials đầu tiên đầy đủ để có dữ liệu cơ sở.
  + n_warmup_steps (25): Mỗi trial phải chạy ít nhất 25 epochs trước khi bị xem xét dừng sớm.
  + interval_steps(3): Thực hiện so sánh sau mỗi 3 epoch sau khi vượt qua n_warmup_steps

Sử dụng SQLite (.db) để lưu trữ lịch sử các lần thử nghiệm
  + Cơ chế tự động copy database từ /input sang /working trên Kaggle giúp duy trì trạng thái huấn luyện ngay cả khi phiên làm việc bị ngắt quãng (load_if_exists=True).
  + Best Model: best_model_trial_{n}.pth - Lưu lại trọng số của mô hình đạt độ chính xác cao nhất trong mỗi trial.
  + Optuna Study: File optuna_study.db chứa toàn bộ lịch sử biến thiên của accuracy theo các bộ tham số.

Hàm loss(Hàm mất mát): Cross Entropy Loss
+ Thay vì ép mô hình hội tụ tuyệt đối vào nhãn đúng, label smoothing phân phối một phần nhỏ xác suất sang các nhãn khác, giúp mô hình không bị overconfident, giảm thiểu Overfitting và giúp mô hình thích nghi tốt hơn với những ảnh có biểu cảm không rõ ràng.

Hàm Tối ưu: AdamW
+ Sử dụng weight decay giúp các trọng số của ResNet18 nhỏ hơn, mượt hơn, từ đó cải thiện khả năng tổng quát hóa trên tập Test một cách đáng kể.

Cosine Annealing Scheduler: Điều khiển learning rate
+ Learning rate không cố định mà biến thiên theo hàm Cosine trong suốt 100 epochs
+ Learning rate giảm dần theo đường cong Cosine về gần mức 0,giúp mô hình "thăm dò" kỹ lưỡng vùng cực tiểu toàn cục, tránh hiện tượng nhảy vọt qua điểm tối ưu ở những epoch cuối

Quy trình Vòng lặp Huấn luyện (Train Loop)
+ Chế độ Train: Kích hoạt model.train(), áp dụng Dropout và tính toán Gradient.
+ Cập nhật Optimizer và Scheduler: Thực hiện optimizer.step() sau mỗi batch và scheduler.step() sau mỗi epoch.
+ Đánh giá (Evaluation): Chuyển sang model.eval() và torch.no_grad() để đo đạc độ chính xác khách quan nhất trên tập Test.
+ Lưu trữ Trọng số: Chỉ lưu lại file .pth khi test_acc vượt qua giá trị cao nhất cũ (max1), đảm bảo bạn luôn có "phiên bản tốt nhất" của trial đó.
+ Cắt tỉa (Pruning): trial.should_prune() sẽ theo dõi test_acc. Nếu sau 25 epochs (warmup) mà kết quả không khả quan so với các trial khác, nó sẽ bị dừng ngay lập tức để tiết kiệm tài nguyên GPU.
