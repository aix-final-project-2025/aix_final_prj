from django.db import models

class RecyclingItem(models.Model): # 세희
    # 파일 경로를 저장합니다.
    image = models.ImageField(upload_to='recycling_images/') 
    
    # 사용자가 입력/수정할 수 있는 설명 필드
    description = models.TextField(blank=True, default="설명을 입력해주세요.") 
    
    # 최신순 정렬을 위한 시간 필드
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # 최신순(내림차순)으로 정렬
        ordering = ['-uploaded_at'] 
        
    def __str__(self):
        return f"Item {self.id} - {self.description[:20]}"