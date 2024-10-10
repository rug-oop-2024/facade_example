from __future__ import annotations

class ParametersDict(dict):
    def _get_keys_as_list(self) -> list:
        return list(self.keys()).sort()
    
    def update(self, new_dict: "ParametersDict") -> None:
        if self._get_keys_as_list() == new_dict._get_keys_as_list() or len(self) == 0:
            super().update(new_dict)
