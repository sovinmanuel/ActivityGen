from chromadb import Client, Settings
from keybert import KeyBERT
import uuid

COLLECTION_NAME = "app_category_collection"
PERSIST_DIR_NAME = "./sim_db"
BASE_CATEGORY_DICT = {
    "Social Networking": "Connect with friends; Social media platform; Sharing photos and videos; News feed; Friend requests; Likes and comments; Messaging; Status updates; Events and invitations; Privacy settings; Profile customization; Groups and communities; Trending topics; Follow/unfollow; Emojis and stickers; Tagging and mentions; Live streaming; News articles; Direct messaging; Video calls",
    "Productivity": "Task management; Calendar integration; Note-taking; Project collaboration; To-do lists; Reminders and notifications; File sharing and syncing; Document editing; Time tracking; Team communication; Cloud storage; Goal setting; Workflow automation; Productivity analytics; Email integration; Offline access; Virtual meetings; Presentation tools; Mind mapping; Password management",
    "Entertainment": "Streaming movies and shows; Music playlists; Podcasts; Gaming; Virtual reality; Live events and concerts; Comedy and stand-up; Celebrity gossip; Movie reviews and ratings; Trending videos; Discover new artists; Personalized recommendations; Create playlists; Podcast subscriptions; Explore genres; Playlists for workouts; Live sports streaming; Radio stations; Augmented reality; User-generated content",
    "Health and Fitness": "Fitness tracking; Exercise routines; Nutrition and meal planning; Calorie counting; Weight management; Sleep tracking; Water intake reminders; Meditation and mindfulness; Workout challenges; Personal trainers; Progress monitoring; Health goals; Healthy recipes; Exercise videos; Heart rate monitoring; GPS tracking; Social support groups; Running and cycling routes; Exercise equipment reviews; Yoga and stretching guides",
    "Travel and Navigation": "Trip planning; Hotel bookings; Flight search and booking; Travel itineraries; Navigation maps; Local attractions; Restaurant recommendations; Public transportation; Offline maps; Car rentals; Travel guides; Currency conversion; Language translation; Sightseeing tours; Weather forecasts; Travel reviews; Visa requirements; Road trip planner; Destination inspiration; Travel insurance",
    "Education": "Language learning; Online courses; Study materials; Exam preparation; Tutoring services; Educational games; Online lectures; Digital textbooks; Virtual classrooms; Study groups; Certification programs; Skill development; Homework help; Subject-specific resources; Flashcards; Educational videos; Quizzes and assessments; Learning analytics; Collaboration tools; Interactive simulations",
    "News and Information": "Breaking news; Headlines; Local news; World news; Business news; Sports updates; Weather forecasts; Politics; Technology news; Science articles; Entertainment news; Lifestyle articles; Health news; Trending stories; Opinion pieces; Investigative journalism; Daily briefing; News analysis; International affairs; Stock market news",
    "Food and Dining": "Recipes; Restaurant reviews; Online food delivery; Meal planning; Cooking tips; Healthy eating; Recipe videos; Wine pairing; Dietary restrictions; Meal tracking; Food blogs; Culinary inspiration; Local food recommendations; Grocery shopping; Farmers markets; Food festivals; Cooking classes; Nutrition information; Specialty diets; Food photography",
    "Shopping": "Online shopping; E-commerce; Fashion trends; Deals and discounts; Customer reviews; Wishlist creation; Price comparison; Shopping apps; Personalized recommendations; Shopping cart; Order tracking; Fashion accessories; Home decor; Electronics; Beauty products; Online marketplaces; Sustainable shopping; Gift ideas; Cashback offers; Shopping rewards; Flash sales",
    "Coding": "Programming languages; Code editors; IDEs; Code snippets; Version control; Debugging tools; API documentation; Stack Overflow; Code review; Software development; Web development; Mobile app development; Backend development; Frontend development; Full-stack development; Code testing; Code optimization; Code libraries; Software architecture; Agile development",
    "Task Management": "Task tracking; Project management; Team collaboration; Task assignments; Task prioritization; Task deadlines; Task reminders; Task progress tracking; Task dependencies; Kanban boards; Gantt charts; Time tracking; Task analytics; Task automation; Task delegation; Task notifications; Task comments; Task attachments; Task history; Task templates",
    "Finance and Investment": "Personal finance; Budget management; Investment strategies; Retirement planning; Stock market analysis; Portfolio management; Wealth management; Tax optimization; Financial goals; Real estate investment; Cryptocurrency trading; Financial news; Economic trends; Financial calculators; Credit score improvement; Loan management; Insurance policies; Tax filing; Financial advisory; Mutual funds",
    "Business and Communication": "Business networking; Professional profiles; Business collaboration; Team communication; Email management; Meeting scheduling; Conference calls; Project management; Business development; CRM software; Sales management; Lead generation; Marketing analytics; Customer support; Employee management; Business news; Market research; Business planning; Business strategy; Remote work",
    "Lifestyle and Fashion": "Fashion trends; Style inspiration; Beauty tips; Wellness practices; Home decor ideas; Self-care routines; Travel inspiration; Food recipes; Fitness motivation; Relationship advice; Personal growth; Parenting tips; Mindfulness practices; Healthy habits; Book recommendations; Pet care; Sustainable living; Fashion accessories; Interior design; Fashion bloggers",
    "File Hosting": "Upload; Download; File Sharing; Cloud Storage; File Transfer; File Management; Share Files; Data Security; Online Collaboration; Access Control; Sync Files; File Backup; Version Control; File Organization; Data Privacy; Remote Access; Document Collaboration; File Hosting Service; Multi-device Syncing; Large File Sharing; Data Accessibility; Dropbox",
}
KEY_EXTRACT_MODEL = KeyBERT()


class CategoryCollection:
    def __init__(self, db_impl="duckdb+parquet"):
        self.db_impl = db_impl
        self.persist_directory = PERSIST_DIR_NAME
        self.client = self._create_client()
        self.collection = self._get_or_create_collection()

    def _create_client(self):
        return Client(
            Settings(
                chroma_db_impl=self.db_impl, persist_directory=self.persist_directory
            )
        )

    def _get_or_create_collection(self):
        existing_collections = self.client.list_collections()
        if not any(
            collection.name == COLLECTION_NAME for collection in existing_collections
        ):
            print("Initializing CategoryCollection")
            collection = self.client.create_collection(name=COLLECTION_NAME)
            for category, keywords in BASE_CATEGORY_DICT.items():
                collection.add(
                    documents=[keywords],
                    metadatas=[{"category": category}],
                    ids=[uuid.uuid4().hex],
                )
            self.client.persist()
            return collection
        else:
            return self.client.get_collection(name=COLLECTION_NAME)

    def add_category(self, category, keywords):
        self.collection.add(
            documents=[keywords],
            metadatas=[{"category": category}],
            ids=[uuid.uuid4().hex],
        )

    def query_categories(self, query_text, n_results=1):
        word_list = query_text.split(";")
        word_count = len(word_list)
        top_n = max(int(word_count * 0.1), 5)
        keywords = KEY_EXTRACT_MODEL.extract_keywords(query_text, top_n=top_n)
        query_text = ""
        for tuple in keywords:
            query_text += tuple[0] + ";"
        return self.collection.query(query_texts=query_text, n_results=n_results)
