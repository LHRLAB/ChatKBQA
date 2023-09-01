from llmtuner import ChatModel


def main():
    chat_model = ChatModel()
    query = "Generate a Logical Form query that retrieves the information corresponding to the given question. \nQuestion: { what does jamaican people speak }"
    output = chat_model.chat_beam(query)
    print(output)


if __name__ == "__main__":
    main()
