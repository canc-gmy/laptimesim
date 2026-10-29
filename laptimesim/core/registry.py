"""Plugin registry system for modular vehicle dynamics subsystems."""

from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T")


class Registry:
    """Generic registry for registering and instantiating modular components."""

    def __init__(self, category: str):
        self.category = category
        self._registry: Dict[str, Type[Any]] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a class under a specific name.

        Args:
            name: Identifier for the model implementation.
        """
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                raise ValueError(
                    f"Model '{name}' is already registered in category '{self.category}'."
                )
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type[Any]:
        """Retrieve a registered class by name."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"Model '{name}' not found in registry '{self.category}'. "
                f"Available models: {available}"
            )
        return self._registry[name]

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Instantiate a registered class by name with provided arguments."""
        cls = self.get(name)
        return cls(*args, **kwargs)

    def list_available(self) -> list[str]:
        """List all registered models in this category."""
        return sorted(list(self._registry.keys()))


class SubsystemRegistry:
    """Global manager for registries across all vehicle subsystems."""

    def __init__(self):
        self._categories: Dict[str, Registry] = {}

    def category(self, category_name: str) -> Registry:
        """Get or create a registry category."""
        if category_name not in self._categories:
            self._categories[category_name] = Registry(category_name)
        return self._categories[category_name]

    def register(self, category: str, name: str) -> Callable[[Type[T]], Type[T]]:
        """Convenience decorator to register a model in a category."""
        return self.category(category).register(name)

    def create(self, category: str, name: str, *args: Any, **kwargs: Any) -> Any:
        """Convenience method to instantiate a registered model."""
        return self.category(category).create(name, *args, **kwargs)

    def list_all(self) -> Dict[str, list[str]]:
        """List all categories and registered models."""
        return {cat: reg.list_available() for cat, reg in self._categories.items()}


# Global singleton instance
model_registry = SubsystemRegistry()
