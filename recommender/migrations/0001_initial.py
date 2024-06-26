# Generated by Django 4.1 on 2024-05-24 05:07

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Poema",
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
                ("titulo", models.CharField(max_length=200)),
                ("poema", models.TextField()),
                ("poeta", models.CharField(max_length=100)),
                ("tags", models.CharField(max_length=200)),
            ],
        ),
    ]
