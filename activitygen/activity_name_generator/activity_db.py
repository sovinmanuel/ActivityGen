import sqlite3


class ActivityNameDatabaseManager:
    def __init__(self, db_name="activity_names.db"):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS app_data
                                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                app_name TEXT,
                                component_class TEXT,
                                inner_text TEXT,
                                activity_names TEXT)"""
        )
        self.connection.commit()

    def insert_data(self, app_name, component_class, inner_text, activity_names):
        if self.check_entry_exists(app_name, component_class, inner_text):
            return
        self.cursor.execute(
            """INSERT INTO app_data
                                (app_name, component_class, inner_text, activity_names)
                                VALUES (?, ?, ?, ?)""",
            (app_name, component_class, inner_text, activity_names),
        )
        self.connection.commit()

    def get_all_data(self):
        self.cursor.execute("""SELECT * FROM app_data""")
        return self.cursor.fetchall()

    def search_by_app_name(self, app_name):
        self.cursor.execute(
            """SELECT * FROM app_data WHERE app_name = ?""", (app_name,)
        )
        return self.cursor.fetchall()

    def update_activity_names(self, app_name, new_activity_names):
        self.cursor.execute(
            """UPDATE app_data SET activity_names = ? WHERE app_name = ?""",
            (new_activity_names, app_name),
        )
        self.connection.commit()

    def delete_data(self, app_name):
        self.cursor.execute("""DELETE FROM app_data WHERE app_name = ?""", (app_name,))
        self.connection.commit()

    def check_entry_exists(self, app_name, component_class, inner_text):
        self.cursor.execute(
            """SELECT activity_names FROM app_data WHERE app_name = ? AND component_class = ? AND inner_text = ?""",
            (app_name, component_class, inner_text),
        )
        result = self.cursor.fetchone()
        if result:
            return result[0]
        else:
            return None

    def close_connection(self):
        self.cursor.close()
        self.connection.close()
