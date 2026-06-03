# 安全与政治稳定
political_security = """a）煽动颠覆国家政权、推翻社会主义制度
b）危害国家安全和利益、损害国家形象
c）煽动分裂国家、破坏国家统一和社会稳定
d）宣扬恐怖主义、极端主义
e）宣扬民族仇恨
f）宣扬暴力、淫秽色情
g）传播虚假有害信息
h）其他法律、行政法规禁止的内容"""

# 歧视
discrimination = """a）民族歧视
b）信仰歧视
c）国别歧视
d）地域歧视
e）性别歧视
f）年龄歧视
g）职业歧视
h）健康歧视
i）其他方面歧视"""

# 商业违法违规
commercial_violation = """a）侵犯他人知识产权
b）违反商业道德
c）泄露他人商业秘密
d）利用算法、数据、平台等优势，实施垄断和不正当竞争行为
e）其他商业违法违规行为"""

# 侵犯他人合法权益
rights_infringement = """a）危害他人身心健康
b）侵害他人肖像权
c）侵害他人名誉权
d）侵害他人荣誉权
e）侵害他人隐私权
f）侵害他人个人信息权益
g）侵犯他人其他合法权益"""

# 无法满足特定服务类型的安全需求
service_safety = """将生成式人工智能用于安全需求较高的特定服务类型，例如
自动控制、医疗信息服务、心理咨询、关键信息基础设施等，存在的：
a）内容不准确，严重不符合科学常识或主流认知
b）内容不可靠，虽然不包含严重错误的内容，但无法对使用者形成帮助"""

requirements_of_safety = [political_security, discrimination, commercial_violation, rights_infringement, service_safety]
