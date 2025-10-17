from django.db import models

class Article(models.Model):
    title = models.CharField(max_length=200, verbose_name="기사 제목")
    content = models.TextField(verbose_name="본문 내용")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title