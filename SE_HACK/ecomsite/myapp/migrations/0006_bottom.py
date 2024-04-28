# Generated by Django 4.2 on 2024-04-27 22:06

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("myapp", "0005_top"),
    ]

    operations = [
        migrations.CreateModel(
            name="Bottom",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "image",
                    models.ImageField(blank=True, null=True, upload_to="bottom_images"),
                ),
                ("gender", models.CharField(default="female", max_length=10)),
                ("color", models.CharField(max_length=100)),
                ("type", models.CharField(max_length=100)),
                ("season", models.CharField(max_length=100)),
                ("length", models.CharField(max_length=100)),
            ],
        ),
    ]
