from abc import ABC, abstractmethod

class AbstractMapper(ABC):
    """탐지 결과를 매핑하기 위한 추상 베이스 클래스입니다."""

    @abstractmethod
    def map(self, detection: dict) -> dict:
        """탐지 데이터를 다른 표현으로 매핑합니다.

        Args:
            detection (dict): 키포인트와 객체 정보를 포함하는 탐지 데이터

        Returns:
            dict: 매핑된 탐지 데이터
        """
        pass