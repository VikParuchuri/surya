'''
  Session封装类
  基于Streamlit API 实现用于存储用户的会话信息
'''
import re


class Session:
    def __init__(self, st, key, value):
        self._st = st
        self._key = key
        self._value = value

    # 获取所有键
    def keys(self):
        return self._st.session_state.keys()
    # 获取 支持正则匹配

    def get(self, str, is_reg=False):
        # 如果开启正则匹配
        if is_reg:
            # 正则
            pattern = re.compile(r"^" + re.escape(str))
            print('pattern', pattern)
            # 满足条件的集合 用于返回所有的键
            result = []
            # 遍历所有的键
            for key in self.keys():
                if pattern.match(key):
                    result.append(self._st.session_state[key])

            print("regex::get", result)
            return result
        # 不开启正则匹配 就直接查找返回
        else:
            if str not in self._st.session_state:
                return None

            return self._st.session_state[f'{str}']

    def set(self, str, value):
        print('session::set', str, value)
        self._st.session_state[f'{str}'] = value

    def remove(self, str):
        if str not in self._st.session_state:
            return

        else:
            self._st.session_state.pop(f'{str}')
