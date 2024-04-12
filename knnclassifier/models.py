from django.db import models

class Addpred(models.Model):
    img_path = models.ImageField(upload_to='images/')  # Or models.FileField() if it's not an image
    prediction = models.TextField()
    print('models img_path', img_path)
    print('models prediction', prediction)
