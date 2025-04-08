from django.db import models
from django.utils import timezone

class Title(models.Model):
    title = models.CharField(max_length=255, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title
    
    class Meta:
        ordering = ['title']

class TitleApplication(models.Model):
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('accepted', 'Accepted'),
        ('rejected', 'Rejected'),
    )
    
    title = models.CharField(max_length=255)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    submission_date = models.DateTimeField(default=timezone.now)
    rejection_reason = models.TextField(blank=True, null=True)
    verification_probability = models.IntegerField(default=0)
    
    def __str__(self):
        return f"{self.title} ({self.status})"
    
    class Meta:
        ordering = ['-submission_date']