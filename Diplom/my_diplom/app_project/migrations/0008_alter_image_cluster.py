# Generated by Django 5.1.4 on 2024-12-22 22:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app_project', '0007_image_cluster'),
    ]

    operations = [
        migrations.AlterField(
            model_name='image',
            name='cluster',
            field=models.IntegerField(blank=True, null=True, verbose_name='Кластер'),
        ),
    ]
