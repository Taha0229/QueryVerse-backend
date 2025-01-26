from query_verse.db.config import SessionLocal, init_db
from query_verse.models.config import User, Product, Order


def seed_data():
    db = SessionLocal()
    init_db()

    users = [
        User(name="Alice", email="alice@example.com"),  # 1
        User(name="Bob", email="bob@example.com"),  # 2
        User(name="Charlie", email="charlie@example.com"),  # 3
        User(name="Diana", email="diana@example.com"),  # 4
        User(name="Eve", email="eve@example.com"),  # 5
        User(name="Fauna", email="fauna@example.com"),  # 6
        User(name="George", email="george@example.com"),  # 7
        User(name="Harry", email="harry@example.com"),  # 8
        User(name="Issac", email="issac@example.com"),  # 9
        User(name="Jack", email="jack@example.com"),  # 10
        User(name="Kris", email="kris@example.com"),  # 11
        User(name="Larry", email="larry@example.com"),  # 12
        User(name="Mary", email="mary@example.com"),  # 13
        User(name="Neo", email="neo@example.com"),  # 14
    ]
    products = [
        Product(name="Apple Iphone 14", price=1999.99),  # 1
        Product(name="Nvidia RTX 5090", price=2099.99),  # 2
        Product(name="Asus TUF A15", price=600.00),  # 3
        Product(name="Sony Alpha 7", price=999.99),  # 4
        Product(name="Kingston HyperX 8 GB RAM", price=100.00),  # 5
        Product(name="JioTag", price=50.00),  # 6
        Product(name="Samsung Galaxy S24 Ultra", price=2000.00),  # 7
        Product(name="OnePlus Nord Buds 2r", price=30.00),  # 8
        Product(name="Legion M600", price=200.00),  # 9
    ]
    orders = [
        Order(user_id=1, product_id=1, quantity=2),
        Order(user_id=1, product_id=3, quantity=1),
        Order(user_id=2, product_id=4, quantity=5),
        Order(user_id=3, product_id=2, quantity=3),
        Order(user_id=3, product_id=5, quantity=1),
        Order(user_id=4, product_id=1, quantity=4),
        Order(user_id=4, product_id=2, quantity=2),
        Order(user_id=5, product_id=3, quantity=1),
        Order(user_id=6, product_id=9, quantity=1),
        Order(user_id=7, product_id=9, quantity=1),
        Order(user_id=7, product_id=8, quantity=4),
        Order(user_id=8, product_id=7, quantity=3),
        Order(user_id=9, product_id=8, quantity=1),
        Order(user_id=10, product_id=2, quantity=2),
        Order(user_id=11, product_id=3, quantity=2),
        Order(user_id=11, product_id=5, quantity=2),
        Order(user_id=12, product_id=9, quantity=1),
        Order(user_id=13, product_id=6, quantity=1),
        Order(user_id=13, product_id=4, quantity=1),
        Order(user_id=14, product_id=3, quantity=1),
    ]

    db.add_all(users)
    db.add_all(products)
    db.add_all(orders)

    db.commit()
    db.close()


if __name__ == "__main__":
    seed_data()
