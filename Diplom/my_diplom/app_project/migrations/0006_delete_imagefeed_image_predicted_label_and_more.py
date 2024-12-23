# Generated by Django 5.1.4 on 2024-12-22 14:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app_project', '0005_imagefeed'),
    ]

    operations = [
        migrations.DeleteModel(
            name='ImageFeed',
        ),
        migrations.AddField(
            model_name='image',
            name='predicted_label',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AlterField(
            model_name='image',
            name='image',
            field=models.ImageField(upload_to='images/', verbose_name='Изображение'),
        ),
    ]