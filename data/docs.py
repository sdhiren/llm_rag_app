from langchain.schema import Document
# Sample data to work with
confluence_docs = [
    Document(page_content="The standard deployment process requires code reviews, CI pipeline execution, and approval from QA before release."),
    Document(page_content="All production deployments must be approved by the release manager and require a rollback plan."),
    Document(page_content="Security processes mandate that all dependencies are scanned weekly using Snyk and reported to the security team."),
]

codebase_docs = [
    Document(page_content="def calculate_discount(price, percentage): return price - (price * percentage/100)"),
    Document(page_content="class ShoppingCart: def __init__(self): self.items = []"),
    Document(page_content="def connect_to_db(uri): # establishes a PostgreSQL connection using psycopg2.connect"),
]