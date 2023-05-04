import tiktoken


def count_n_tokens(text):
    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text))


def split_section(text: str, max_tokens: int = 0, prompt_n_tokens: int = 0):
    """
    Divide a piece of text to paragraphs each paragraph's length is shorter than the max_token attribute
    :param text:
    :param max_tokens:
    :param prompt_n_tokens:
    :return:
    """
    text_list = text.split('.')
    result_list = []
    section = ''
    for txt in text_list:
        txt = txt.strip()
        # Check for empty strings
        if not txt or count_n_tokens(txt) == 1:
            continue

        # Check if the len of the current section is less than the max_number of tokens
        if count_n_tokens(section) + count_n_tokens(txt) + prompt_n_tokens < max_tokens:
            section += txt + " "

        else:

            result_list.append(section)
            section = txt +" "

    # To capture the last part of the section
    if section:
        result_list.append(section)

    return result_list


def split_text(text: str, max_tokens: int = 0, prompt_n_tokens: int = 0):
    """
    Split the text (terms & agreements) into sections, if the section length is larger than the max tokens it is divided
    into smaller chunks
    :param text:
    :param max_tokens:
    :param prompt_n_tokens:
    :return:
    """
    text_list = text.split('\n')
    result = []
    section = ""
    for txt in text_list:
        txt = txt.strip()
        if not txt:
            continue

        if count_n_tokens(section) + count_n_tokens(txt) + prompt_n_tokens < max_tokens:
            section += txt + "\n"

        else:
            # Part of the text is too long
            if count_n_tokens(txt) + prompt_n_tokens >= max_tokens:
                result += split_section(text=txt, max_tokens=max_tokens, prompt_n_tokens=prompt_n_tokens)
            # Section limit is reached
            else:
                result.append(section)
                section = txt
    result.append(section)
    return result
