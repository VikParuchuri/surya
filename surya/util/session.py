'''
  Session封装类
  基于Streamlit API 实现用于存储用户的会话信息
'''


class Session:
    def __init__(self, st, key, value):
        self._st = st
        self._key = key
        self._value = value

    def get(self, str):

        if str not in self._st.session_state:
            return None

        else:
            return self._st.session_state[f'{str}']

    def set(self, str, value):
        self._st.session_state[f'{str}'] = value

    def remove(self, str):
        if str not in self._st.session_state:
            return

        else:
            self._st.session_state.pop(f'{str}')
