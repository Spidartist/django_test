from django.contrib import admin
from django.urls import path
from django.conf.urls import include
from django.urls import re_path as url
from rest_framework.routers import DefaultRouter
from server.app.views import EndpointViewSet
from server.app.views import MLModelViewSet
from server.app.views import MLModelStatusViewSet
from server.app.views import MLRequestViewSet


router = DefaultRouter(trailing_slash=False)
router.register(r"endpoints", EndpointViewSet, basename="endpoints")
router.register(r"mlmodels", MLModelViewSet, basename="mlmodels")
router.register(r"mlmodelstatuses", MLModelStatusViewSet,
                basename="mlmodelstatuses")
router.register(r"mlrequests", MLRequestViewSet, basename="mlrequests")

urlpatterns = [
    url(r"^api/v1/", include(router.urls)),
    path('admin/', admin.site.urls),
]
