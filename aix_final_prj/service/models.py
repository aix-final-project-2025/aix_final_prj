from django.db import models

class GroupCode(models.Model):
    """
    분류 코드 사전 테이블
    """
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=200, verbose_name='코드명(한글)', unique=True)
    code = models.CharField(max_length=50, verbose_name='기존 코드명', unique=True)
    numeric_code = models.CharField(max_length=5, verbose_name='문자형 숫자 코드', unique=True, editable=False)
    desc = models.TextField(blank=True, null=True, verbose_name='설명')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='등록일시')

    class Meta:
        db_table = 'group_code'
        verbose_name = '분류 코드'
        verbose_name_plural = '분류 코드 목록'
        ordering = ['id']

    def __str__(self):
        return f"{self.name} ({self.numeric_code})"

    def save(self, *args, **kwargs):
        if not self.numeric_code:
            from random import randint
            while True:
                code = f"{randint(0, 99999):05d}"
                if not GroupCode.objects.filter(numeric_code=code).exists():
                    self.numeric_code = code
                    break
        super().save(*args, **kwargs)

class RecyclableResult(models.Model):
    """
    분류 결과 테이블
    """
    group_code = models.ForeignKey(
        GroupCode,
        on_delete=models.CASCADE,
        related_name='results',
        verbose_name='분류 코드'
    )

    PREDICTED_CLASS = models.CharField(max_length=100, verbose_name='생활쓰레기 구분', null=True, blank=True)
    CONFIDENCE = models.FloatField(verbose_name='신뢰도', null=True, blank=True)
    RESULT_MESSAGE = models.TextField(verbose_name='결과 메시지', null=True, blank=True)
    CONFIDENCE_LEVEL = models.CharField(max_length=50, verbose_name='신뢰도 등급', null=True, blank=True)
    TOP_3 = models.TextField(verbose_name='TOP 3 결과', null=True, blank=True)
    RECYCLING_GUIDE = models.TextField(verbose_name='처리 가이드', null=True, blank=True)
    RESULT_IMAGE = models.ImageField(upload_to='recyclable_results/', verbose_name='결과 이미지', null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True, verbose_name='등록일시')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='수정일시')

    class Meta:
        db_table = 'recyclable_result'
        verbose_name = '분류 결과'
        verbose_name_plural = '분류 결과 목록'
        ordering = ['created_at']

    def __str__(self):
        return f"[{self.group_code.name}] {self.PREDICTED_CLASS or '결과 없음'}"
    



class CountryPf(models.Model):
    COUNTRY_CHOICES = [
        ('KR', '대한민국'),
        ('US', '미국'),
        ('ES', '스페인'),
        # 필요에 따라 추가
    ]
    GENDER_CHOICES = [
        ('M', '남성'),
        ('F', '여성'),
        ('O', '기타'),
    ]

    country = models.CharField(max_length=2, choices=COUNTRY_CHOICES, verbose_name='국가')
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, verbose_name='성별')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='등록일')

    class Meta:
        verbose_name = '사용자 프로필'
        verbose_name_plural = '사용자 프로필 목록'
        db_table = 'user_profile'
        ordering = ['created_at']

    def __str__(self):
        return f"{self.country}-{self.gender}-{self.created_at.strftime('%Y-%m-%d')}"    