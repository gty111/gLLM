from gllm.entrypoints.protocol import (ChatCompletionStreamResponse, ChatCompletionResponseStreamChoice, 
                                       DeltaMessage, ChatCompletionRequest, ChatCompletionResponseChoice, 
                                       ChatCompletionResponse, ChatMessage, UsageInfo)
from gllm.async_llm_engine import AsyncStream


async def chat_completion_generator(stream: AsyncStream, request: ChatCompletionRequest):
    full_text = ''
    async for text in stream:
        full_text += text
    choice_data = ChatCompletionResponseChoice(index=0,
                                               message=ChatMessage(role='assistant',
                                                                   content=full_text))
    response = ChatCompletionResponse(choices=[choice_data],
                                      usage=UsageInfo(),
                                      model=request.model)
    return response


async def chat_completion_stream_generator(stream: AsyncStream, request: ChatCompletionRequest):
    async for text in stream:
        choice_data = ChatCompletionResponseStreamChoice(index=0,
                                                         delta=DeltaMessage(
                                                             content=text))
        chunk = ChatCompletionStreamResponse(choices=[choice_data],
                                             model=request.model)
        data = chunk.model_dump_json(exclude_unset=True)
        yield f'data: {data}\n\n'
    yield "data: [DONE]\n\n"
