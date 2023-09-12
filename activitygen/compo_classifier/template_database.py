from peewee import *
from PIL import Image
import os


class TemplateDatabase:
    def __init__(self, db_file="templates.db"):
        self.db = SqliteDatabase(db_file)

    def initialize(self):
        if not os.path.exists(self.db.database):
            with self.db:
                self.db.create_tables([Template])
        else:
            self.db.connect()

    def add_template(self, name, image_path, matching_method):
        template = Template(
            name=name, image_path=image_path, matching_method=matching_method
        )
        template.save()

    def find_template_by_name(self, name):
        try:
            template = Template.get(Template.name == name)
            return template
        except Template.DoesNotExist:
            return None


class Template(Model):
    name = CharField()
    image_path = CharField(unique=True)
    matching_method = CharField()

    class Meta:
        database = SqliteDatabase(None)


class ImageUtils:
    @staticmethod
    def save_image(image_path, image):
        directory = os.path.dirname(image_path)
        os.makedirs(directory, exist_ok=True)
        image.save(image_path)

    @staticmethod
    def load_image(image_path):
        return Image.open(image_path)
