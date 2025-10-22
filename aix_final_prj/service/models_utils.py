from .models import CountryPf
from django.core.exceptions import ObjectDoesNotExist

def get_profile_setting(country: str, gender: str, active: bool = None):
    """
    국가, 성별, 그리고 사용 여부(active)를 기준으로 CountryPf 레코드를 조회하는 공통 함수입니다.
    
    Args:
        country (str): 조회할 국가 코드.
        gender (str): 조회할 성별 코드.
        active (bool, optional): 조회할 사용 여부 상태 (True/False). 
                                 None이면 active 상태와 무관하게 조회합니다.
        
    Returns:
        CountryPf: 조회된 객체. (없으면 None)
    """
    
    filter_kwargs = {
        'country': country,
        'gender': gender,
    }
    
    # active 값이 명시적으로 전달된 경우에만 필터 조건에 추가
    if active is not None:
        filter_kwargs['active'] = active
    
    try:
        # filter_kwargs에 포함된 모든 조건을 만족하는 레코드 하나를 조회합니다.
        profile = CountryPf.objects.get(**filter_kwargs)
        return profile
    except ObjectDoesNotExist:
        # 레코드가 없는 경우 None을 반환합니다.
        return None