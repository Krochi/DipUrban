# Generated by Django 5.1.4 on 2024-12-22 12:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app_project', '0004_image_predicted_class'),
    ]

    operations = [
        migrations.CreateModel(
            name='ImageFeed',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='uploads/')),
                ('predicted_class', models.CharField(blank=True, max_length=255, null=True)),
                ('predicted_label', models.CharField(blank=True, max_length=255, null=True)),
            ],
        ),
    ]
