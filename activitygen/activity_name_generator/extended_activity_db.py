from peewee import *
import os

db = SqliteDatabase("activity_names_extended.db")


class AppData(Model):
    app_name = CharField()
    app_category = CharField()
    component_class = CharField()
    inner_text = TextField()
    surrounding_text = TextField()
    activity_names = TextField()

    class Meta:
        database = db


class ActivityNameDatabaseManagerExtended:
    def __init__(self):
        if not os.path.exists(db.database):
            db.connect()
            db.create_tables([AppData])
        else:
            db.init("activity_names_extended.db")
            db.connect()

    def insert_data(
        self,
        app_name,
        app_category,
        component_class,
        inner_text,
        surrounding_text,
        activity_names,
    ):
        AppData.create(
            app_name=app_name,
            app_category=app_category,
            component_class=component_class,
            inner_text=inner_text,
            surrounding_text=surrounding_text,
            activity_names=activity_names,
        )

    def get_all_data(self):
        return AppData.select()

    def search_by_app_name(self, app_name):
        return AppData.select().where(AppData.app_name == app_name)

    def update_activity_names(self, app_name, new_activity_names):
        query = AppData.update(activity_names=new_activity_names).where(
            AppData.app_name == app_name
        )
        query.execute()

    def delete_data(self, app_name):
        query = AppData.delete().where(AppData.app_name == app_name)
        query.execute()

    def check_entry_exists(
        self, app_name, app_category, component_class, inner_text, surrounding_text
    ):
        result = (
            AppData.select()
            .where(
                (AppData.app_name == app_name)
                & (AppData.app_category == app_category)
                & (AppData.component_class == component_class)
                & (AppData.inner_text == inner_text)
                & (AppData.surrounding_text == surrounding_text)
            )
            .first()
        )
        if result:
            return result.activity_names
        else:
            return None

    def close_connection(self):
        db.close()
