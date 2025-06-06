import re

cbn4 = [{"content_type":"file_message","content":"jsconfig.json"},{"content_type":"text_message","content":"[jsconfig.json]这是什么"}]
         
def user_input_paser(user_input):
    text = ''
    for inupt in user_input:
        if inupt['content_type'] == 'text_message':
            text = inupt["content"]
            break
    return re.sub(r"\[.*?\]", "", text)

print(user_input_paser(cbn4))