from fastapi.middleware.cors import CORSMiddleware

def setup_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
# # Chứa cors api
# Các thông số chính thường cấu hình trong file này bao gồm:

# Allowed Origins: Danh sách các domain được phép truy cập (ví dụ: ['http://localhost:3000', 'https://myapps.com']).

# Allowed Methods: Các phương thức được phép như GET, POST, PUT, DELETE.

# Allowed Headers: Các header tùy chỉnh mà frontend có thể gửi lên (như Authorization để gửi token).