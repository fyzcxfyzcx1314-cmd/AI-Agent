# 运行代码： streamlit run ./rag/deepseek_rag/rag_frontend.py
import streamlit as st
import requests
import json
import time
from streamlit_option_menu import option_menu

# 页面设置
st.set_page_config(
    page_title="智能知识助手",
    page_icon="🏭",
    layout="wide"
)

# 后端API配置
BACKEND_URL = "http://127.0.0.1:8000"

# 初始化会话状态，session_state保存上下文会话状态，用户信息，聊天历史，当前选中的标签
if 'user_ctx' not in st.session_state:
    st.session_state.user_ctx = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "知识助手"


def knowledge_assistant():
    st.header("🏭 智能知识助手")

    # 聊天历史显示
    chat_container = st.container(height=500)
    with chat_container:
        for idx, msg in enumerate(st.session_state.chat_history):
            if msg["role"] == "user":
                with st.chat_message("user", avatar="🧑‍🔧"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    # 添加来源标识
                    source_type = "deepseek.pdf"
                    st.caption(f"来源: {source_type}")
                    # 大模型回复的答案
                    st.markdown(msg["content"])
                    # 更健壮的空值检查
                    if msg.get("source_data"):
                        with st.expander("查看来源数据"):
                                    for source in msg.get("source_data"):
                                        st.json(source)
                                        # 安全显示结果
                                        result = source
                                        if result:
                                            try:
                                                # 尝试解析为JSON
                                                result_data = json.loads(result)
                                                st.json(result_data)
                                            except:
                                                # 如果不是JSON，直接显示文本
                                                st.text(result)
                    else:
                        st.warning("无结果数据")

    # 用户输入
    user_input = st.chat_input("请输入您的问题...")

    if user_input:
        cleaned_input = user_input.strip()
        cleaned_input = cleaned_input.replace("'", "")

        # 显示用户消息
        with chat_container:
            with st.chat_message("user", avatar="🧑‍🔧"):
                st.markdown(user_input)

        # 调用后端API
        with st.spinner("正在思考..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={
                        "question": cleaned_input,
                        "user_ctx": st.session_state.user_ctx,
                        "chat_history":st.session_state.chat_history
                    }
                )
                # 添加用户消息到历史
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                # 判断响应状态
                if response.status_code == 200:
                    result = response.json()
                    print(f"result:{result}")

                    # 添加助手回复到历史
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "source_data": result.get("source_data", [])
                    })

                    # 显示助手回复
                    with chat_container:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.markdown(result["answer"])
                            if "source_data" in result:
                                with st.expander("查看来源数据"):
                                    for source in result["source_data"]:
                                        st.json(source)
                else:
                    # st.error(f"请求失败: {response.text}")
                    # 更健壮的错误处理
                    try:
                        error_detail = response.json().get('detail', response.text)
                    except:
                        error_detail = response.text[:500]  # 限制长度防止显示问题

                    # 更友好的错误提示
                    error_msg = f"请求失败: {error_detail}"
                    st.error(error_msg)

                    # 添加到聊天历史
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"抱歉，处理您的请求时出错: {error_msg}"
                    })
            except Exception as e:
                st.error(f"发生错误: {str(e)}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"抱歉，处理您的请求时出错: {str(e)}"
                })


def login_page():
    st.title("智能知识助手登录")
    st.write("请使用您的账户登录")

    with st.form("login_form"):

        role = st.selectbox("角色", ["操作员", "工程师", "专家"], index=1)
        username = st.text_input("用户名", value="engineer_li")
        password = st.text_input("密码", type="password", value="securepass123")

        submitted = st.form_submit_button("登录")
        if submitted:
            if username and password:
                st.session_state.user_ctx = {
                    "role": role,
                    "username": username
                }
                st.session_state.chat_history = []
                st.success("登录成功！正在跳转主界面...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("用户名和密码不能为空")

def main_page():

    # 租户信息显示
    st.sidebar.subheader(f"当前用户")
    st.sidebar.markdown(f"**用户**: {st.session_state.user_ctx['username']}")
    st.sidebar.markdown(f"**角色**: {st.session_state.user_ctx['role']}")

    st.sidebar.divider()
    st.sidebar.markdown("### 快捷操作")
    if st.sidebar.button("清除聊天记录"):
        st.session_state.chat_history = []
        st.rerun()

    # 功能区域
    if st.session_state.selected_tab == "知识助手":
        knowledge_assistant()

# 应用路由
if 'user_ctx' in st.session_state and st.session_state.user_ctx:
    main_page()
else:
    login_page()