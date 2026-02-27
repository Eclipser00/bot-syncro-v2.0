"""Registro de estrategias y asignación de Magic Numbers.

Los Magic Numbers son identificadores numéricos únicos que se envían al broker
con cada orden. Esto permite identificar de forma robusta qué estrategia abrió
cada posición, incluso si los comentarios son truncados o modificados.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Gestiona el registro de estrategias y asigna Magic Numbers únicos.
    
    El Magic Number se genera mediante un hash del nombre de la estrategia,
    garantizando que sea consistente entre ejecuciones del bot.
    """
    
    def __init__(self) -> None:
        self._strategy_to_magic: Dict[str, int] = {}
        self._magic_to_strategy: Dict[int, str] = {}
    
    def register_strategy(self, strategy_name: str) -> int:
        """Registra una estrategia y devuelve su Magic Number único.
        
        Args:
            strategy_name: Nombre de la estrategia a registrar.
            
        Returns:
            Magic Number asignado (entero de 32 bits).
        """
        if strategy_name in self._strategy_to_magic:
            return self._strategy_to_magic[strategy_name]
        
        # Generar Magic Number mediante hash del nombre
        # Usar solo los primeros 8 caracteres del hash para obtener un entero de 32 bits
        hash_hex = hashlib.md5(strategy_name.encode('utf-8')).hexdigest()[:8]
        magic_number = int(hash_hex, 16) % (2**31)  # Mantener en rango de int32 positivo
        
        # Verificar colisiones (muy improbable pero posible)
        if magic_number in self._magic_to_strategy:
            logger.warning(
                "Colisión de Magic Number detectada para %s y %s. Incrementando.",
                strategy_name,
                self._magic_to_strategy[magic_number]
            )
            # Resolver colisión incrementando
            while magic_number in self._magic_to_strategy:
                magic_number = (magic_number + 1) % (2**31)
        
        self._strategy_to_magic[strategy_name] = magic_number
        self._magic_to_strategy[magic_number] = strategy_name
        
        logger.info("Estrategia '%s' registrada con Magic Number: %d", strategy_name, magic_number)
        return magic_number
    
    def get_magic_number(self, strategy_name: str) -> int | None:
        """Obtiene el Magic Number de una estrategia registrada.
        
        Args:
            strategy_name: Nombre de la estrategia.
            
        Returns:
            Magic Number si está registrada, None en caso contrario.
        """
        return self._strategy_to_magic.get(strategy_name)
    
    def get_strategy_name(self, magic_number: int) -> str | None:
        """Obtiene el nombre de estrategia asociado a un Magic Number.
        
        Args:
            magic_number: Magic Number a buscar.
            
        Returns:
            Nombre de la estrategia si existe, None en caso contrario.
        """
        return self._magic_to_strategy.get(magic_number)
    
    def is_registered(self, strategy_name: str) -> bool:
        """Verifica si una estrategia está registrada.
        
        Args:
            strategy_name: Nombre de la estrategia.
            
        Returns:
            True si está registrada, False en caso contrario.
        """
        return strategy_name in self._strategy_to_magic

