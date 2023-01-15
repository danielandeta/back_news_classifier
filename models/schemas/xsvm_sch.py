from pydantic import BaseModel
from typing import List, Any


class XSVMC_BM(BaseModel):
    data:  str #List[conlist(float, min_items=4, max_items=4)]


class xSVMCResponse(BaseModel):
    prediction:  str #List[int]
    ruta_mi1: str
    ruta_mi2: str