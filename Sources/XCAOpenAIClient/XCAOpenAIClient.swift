import Foundation
import OpenAPIRuntime
import OpenAPIURLSession

public struct OpenAIClient {
    
    public let client: Client
    private let urlSession = URLSession.shared
    private let apiKey: String
    
    public init(apiKey: String) {
        self.client = Client(
            serverURL: try! Servers.server1(),
            transport: URLSessionTransport(),
            middlewares: [AuthMiddleware(apiKey: apiKey)])
        self.apiKey = apiKey
    }

    public func promptChatGPT4oMini(
        prompt: String,
        assistantPrompt: String = "You are a helpful assistant",
        responseFormatType: String? = nil, // Accept a string
        prevMessages: [Components.Schemas.ChatCompletionRequestMessage] = []
    ) async throws -> String {
        return try await promptChatGPT(
            prompt: prompt,
            model: .gpt_hyphen_4o_hyphen_mini,
            assistantPrompt: assistantPrompt,
            responseFormatType: responseFormatType,
            prevMessages: prevMessages)
    }
    
    public func promptChatGPT(
        prompt: String,
        model: Components.Schemas.CreateChatCompletionRequest.modelPayload.Value2Payload = .gpt_hyphen_4o,
        assistantPrompt: String = "You are a helpful assistant",
        responseFormatType: String? = nil, // Accept a string
        prevMessages: [Components.Schemas.ChatCompletionRequestMessage] = []
    ) async throws -> String {

        // Build the response_format object if responseFormatType is provided
        var responseFormat: Components.Schemas.CreateChatCompletionRequest.response_formatPayload? = nil
        if let formatType = responseFormatType {
            responseFormat = Components.Schemas.CreateChatCompletionRequest.response_formatPayload(_type: .init(rawValue: formatType))
        }

        // Build the body for the request
        let requestBody = Components.Schemas.CreateChatCompletionRequest(
            messages: [.ChatCompletionRequestAssistantMessage(.init(content: assistantPrompt, role: .assistant))]
            + prevMessages
            + [.ChatCompletionRequestUserMessage(.init(content: .case1(prompt), role: .user))],
            model: .init(value1: nil, value2: model),
            response_format: responseFormat // Use the built response format
        )
        
        let response = try await client.createChatCompletion(body: .json(requestBody))

        switch response {
        case .ok(let body):
            let json = try body.body.json
            guard let content = json.choices.first?.message.content else {
                throw "No Response"
            }
            return content
        case .undocumented(let statusCode, let payload):
            throw "OpenAIClientError - statuscode: \(statusCode), \(payload)"
        }
    }

    public func streamingGenerateSpeechFrom(
        input: String,
        model: Components.Schemas.CreateSpeechRequest.modelPayload.Value2Payload = .tts_hyphen_1,
        voice: Components.Schemas.CreateSpeechRequest.voicePayload = .fable,
        format: Components.Schemas.CreateSpeechRequest.response_formatPayload = .aac
    ) async throws -> AsyncThrowingStream<Data, Error> {
        guard let url = URL(string: "https://api.openai.com/v1/audio/speech") else {
            throw URLError(.badURL)
        }
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let jsonBody: [String: Any] = [
            "model": model.rawValue,  // e.g. "tts-1"
            "input": input,
            "voice": voice.rawValue,
            "response_format": format.rawValue,
            "stream": true
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: jsonBody)

        let (asyncBytes, response) = try await URLSession.shared.bytes(for: request)
        guard let httpResponse = response as? HTTPURLResponse,
              200..<300 ~= httpResponse.statusCode else {
            throw URLError(.badServerResponse)
        }
        
        // Stream chunks (1 KB at a time in this example).
        return AsyncThrowingStream { continuation in
            Task {
                var buffer = Data()
                let chunkSize = 1024
                do {
                    for try await byte in asyncBytes {
                        buffer.append(byte)
                        if buffer.count >= chunkSize {
                            continuation.yield(buffer)
                            buffer.removeAll(keepingCapacity: true)
                        }
                    }
                    // Yield any leftover bytes
                    if !buffer.isEmpty {
                        continuation.yield(buffer)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }
    
    public func generateSpeechFrom(input: String,
                                   model: Components.Schemas.CreateSpeechRequest.modelPayload.Value2Payload = .tts_hyphen_1,
                                   voice: Components.Schemas.CreateSpeechRequest.voicePayload = .fable,
                                   format: Components.Schemas.CreateSpeechRequest.response_formatPayload = .aac
    ) async throws -> Data {
        let response = try await client.createSpeech(body: .json(
            .init(
                model: .init(value1: nil, value2: model),
                input: input,
                voice: voice,
                response_format: format
            )))
        
        switch response {
        case .ok(let response):
            switch response.body {
            case .any(let body):
                var data = Data()
                for try await byte in body {
                    data.append(contentsOf: byte)
                }
                return data
            }
            
        case .undocumented(let statusCode, let payload):
            throw "OpenAIClientError - statuscode: \(statusCode), \(payload)"
        }
    }

    /// Use URLSession manually until swift-openapi-runtime support MultipartForm
    public func generateAudioTransciptions(audioData: Data, fileName: String = "recording.m4a", prompt: String = "", languageCode: String? = nil) async throws -> String {
        var request = URLRequest(url: URL(string: "https://api.openai.com/v1/audio/transcriptions")!)
        let boundary: String = UUID().uuidString
        request.timeoutInterval = 30
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var entries: [MultipartFormDataEntry] = [
            .file(paramName: "file", fileName: fileName, fileData: audioData, contentType: "audio/mpeg"),
            .string(paramName: "model", value: "gpt-4o-transcribe"),
            .string(paramName: "response_format", value: "text"),
            .string(paramName: "prompt", value: prompt)
        ]
        if let languageCode = languageCode, !languageCode.isEmpty {
            entries.append(.string(paramName: "language", value: languageCode))
        }

        let bodyBuilder = MultipartFormDataBodyBuilder(boundary: boundary, entries: entries)

        request.httpBody = bodyBuilder.build()
        let (data, resp) = try await urlSession.data(for: request)
        guard let httpResp = resp as? HTTPURLResponse, httpResp.statusCode == 200 else {
            throw "Invalid Status Code \((resp as? HTTPURLResponse)?.statusCode ?? -1)"
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw "Invalid format"
        }
        
        return text
    }
    
    public func generateDallE3Image(prompt: String,
                                    quality: Components.Schemas.CreateImageRequest.qualityPayload = .standard,
                                    responseFormat: Components.Schemas.CreateImageRequest.response_formatPayload = .url,
                                    style: Components.Schemas.CreateImageRequest.stylePayload = .vivid
                                    
    ) async throws -> Components.Schemas.Image {
        
        let response = try await client.createImage(.init(body: .json(
            .init(
                prompt: prompt,
                model: .init(value1: nil, value2: .dall_hyphen_e_hyphen_3),
                n: 1,
                quality: quality,
                response_format: responseFormat,
                size: ._1024x1024,
                style: style
            ))))
        
        switch response {
        case .ok(let response):
            switch response.body {
            case .json(let imageResponse) where imageResponse.data.first != nil:
                return imageResponse.data.first!
                
            default:
                throw "Unknown response"
            }
            
        case .undocumented(let statusCode, let payload):
            throw "OpenAIClientError - statuscode: \(statusCode), \(payload)"
        }
    }
    
    public func promptChatGPTVision(
        imageData: Data,
        detail: Components.Schemas.ChatCompletionRequestMessageContentPartImage.image_urlPayload.detailPayload = .low,
            maxTokens: Int? = 300) async throws -> String {
        let response = try await client.createChatCompletion(body: .json(.init(
            messages: [
                .ChatCompletionRequestUserMessage(.init(content: .case1("Describe this image in details, provide all visual representations, you can ignore text within the image"), role: .user)),
                       .ChatCompletionRequestUserMessage(
                        .init(content: .case2([.ChatCompletionRequestMessageContentPartImage(
                            .init(_type: .image_url, image_url:
                                    .init(url: "data:image/jpeg;base64,\(imageData.base64EncodedString())", detail: detail)))]
                        ), role: .user))
            ],
            model: .init(value1: nil, value2: .gpt_hyphen_4_hyphen_vision_hyphen_preview),
            max_tokens: maxTokens)))
            
        switch response {
        case .ok(let body):
            let json = try body.body.json
            guard let content = json.choices.first?.message.content else {
                throw "No Response"
            }
            return content
        case .undocumented(let statusCode, let payload):
            throw "OpenAIClientError - statuscode: \(statusCode), \(payload)"
        }
        
    }
    
}
