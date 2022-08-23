import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()

# ML registry
from ml.registry import MLRegistry
from ml.classifier.random_forest import RandomForestClassifier

try:
    # create ML registry
    registry = MLRegistry()
    # Random Forest classifier
    rf = RandomForestClassifier()
    # add to ML registry
    registry.add_algorithm(endpoint_name="classifier",
                            algorithm_object=rf,
                            algorithm_name="random forest",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="Kami",
                            algorithm_description="Random Forest with simple pre- and post-processing")
except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))
