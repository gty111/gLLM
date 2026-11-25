//! `POST /v1/responses`, served through the chat stack.
//!
//! Input items become chat messages; Responses tools (functions, custom tools,
//! namespaces, client tool search) become chat function tools; the chat event
//! stream is turned back into Responses output items / SSE events.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use base64::Engine as _;
use futures::{Stream, StreamExt as _};
use serde_json::{Map, Value, json};
use vllm_chat::{AssistantBlockKind, AssistantContentBlock, ChatEvent, FinishReason};

use super::chat::{finish_error, submit_error};
use super::types::ChatCompletionRequest;
use super::{ApiError, AppState, unix_now};

const TOOL_SEARCH_NAME: &str = "tool_search";
const MAX_INLINE_FILE_BYTES: usize = 50 * 1024 * 1024;
const TEXT_EXTENSIONS: &[&str] = &[
    "c", "cc", "cpp", "css", "csv", "go", "h", "hpp", "html", "ini", "java", "js", "json", "jsx", "log",
    "md", "py", "rb", "rs", "sh", "sql", "toml", "ts", "tsx", "txt", "xml", "yaml", "yml",
];

fn rid() -> String {
    uuid::Uuid::new_v4().simple().to_string()
}

fn bad(param: impl Into<String>, message: impl Into<String>) -> ApiError {
    let param = param.into();
    ApiError::invalid(message, Some(&param), "unsupported_parameter")
}

// ---------------------------------------------------------------------------
// Stored responses (`store` / `previous_response_id`)
// ---------------------------------------------------------------------------

/// Stored entries (with their creation time) and their LRU order.
type Entries = (HashMap<String, (Instant, Value)>, VecDeque<String>);

pub struct ResponseStore {
    entries: Mutex<Entries>,
    max_entries: usize,
    ttl: Duration,
}

impl Default for ResponseStore {
    fn default() -> Self {
        Self { entries: Mutex::new((HashMap::new(), VecDeque::new())), max_entries: 1000, ttl: Duration::from_secs(86400) }
    }
}

impl ResponseStore {
    fn put(&self, id: String, value: Value) {
        let mut guard = self.entries.lock().unwrap();
        let (map, order) = &mut *guard;
        let ttl = self.ttl;
        map.retain(|_, (t, _)| t.elapsed() < ttl);
        order.retain(|k| map.contains_key(k) && k != &id);
        map.insert(id.clone(), (Instant::now(), value));
        order.push_back(id);
        while order.len() > self.max_entries {
            if let Some(old) = order.pop_front() {
                map.remove(&old);
            }
        }
    }

    fn get(&self, id: &str) -> Option<Value> {
        let mut guard = self.entries.lock().unwrap();
        let (map, order) = &mut *guard;
        let (created, value) = map.get(id)?.clone();
        if created.elapsed() >= self.ttl {
            map.remove(id);
            return None;
        }
        order.retain(|k| k != id);
        order.push_back(id.to_owned());
        Some(value)
    }
}

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

/// Callable tool name -> (tool definition, namespace).
type Specs = BTreeMap<String, (Value, Option<String>)>;

fn validate_client_search_item(item: &Value, param: &str) -> Result<(), ApiError> {
    if item["execution"] != "client" {
        return Err(bad(param, "Only client-executed tool search is supported."));
    }
    if item["call_id"].as_str().is_none_or(str::is_empty) {
        return Err(bad(format!("{param}.call_id"), "Tool search items require a nonempty call_id."));
    }
    if item.get("status").and_then(Value::as_str).unwrap_or("completed") != "completed" {
        return Err(bad(format!("{param}.status"), "Tool search history must be completed."));
    }
    Ok(())
}

fn check_grammar(fmt: &Value) -> Result<(), String> {
    match fmt["syntax"].as_str() {
        Some("regex") => regex::Regex::new(fmt["definition"].as_str().unwrap_or_default()).map(|_| ()).map_err(|e| e.to_string()),
        Some("lark") => Ok(()),
        _ => Err("Unsupported custom tool grammar syntax.".into()),
    }
}

fn collect_specs(tools: &[Value], root: &str, loaded: bool) -> Result<Specs, ApiError> {
    fn add(result: &mut Specs, tool: &Value, param: String, namespace: Option<&str>, loaded: bool) -> Result<(), ApiError> {
        if !tool.is_object() {
            return Err(bad(param, "Tools must be objects."));
        }
        let kind = tool["type"].as_str();
        if kind == Some("tool_search") {
            if namespace.is_some() || loaded {
                return Err(bad(param, "tool_search must be a top-level request tool."));
            }
            if tool["execution"] != "client" {
                return Err(bad(param, "Only client-executed tool search is supported."));
            }
            if tool["parameters"]["type"] != "object" {
                return Err(bad(format!("{param}.parameters"), "Client tool search requires an object argument schema."));
            }
            if result.contains_key(TOOL_SEARCH_NAME) {
                return Err(bad(param, format!("Ambiguous tool name: {TOOL_SEARCH_NAME}.")));
            }
            result.insert(TOOL_SEARCH_NAME.into(), (tool.clone(), None));
            return Ok(());
        }
        let Some(name) = tool["name"].as_str().filter(|n| !n.is_empty()) else {
            return Err(bad(param, "Tool names must be nonempty strings."));
        };
        if kind == Some("namespace") && namespace.is_none() {
            if !tool["description"].is_string() {
                return Err(bad(param, "A namespace must contain a description string."));
            }
            let Some(children) = tool["tools"].as_array() else {
                return Err(bad(param, "A namespace must contain a tools array."));
            };
            for (i, child) in children.iter().enumerate() {
                add(result, child, format!("{param}.tools.{i}"), Some(name), loaded)?;
            }
            return Ok(());
        }
        if !matches!(kind, Some("function") | Some("custom")) {
            return Err(bad(param, "Only function/custom tools and their namespaces are supported."));
        }
        let native = namespace.map_or_else(|| name.to_owned(), |ns| format!("{ns}.{name}"));
        if result.contains_key(&native) {
            return Err(bad(param, format!("Ambiguous tool name: {native}.")));
        }
        if kind == Some("custom") {
            let fmt = tool.get("format").cloned().unwrap_or(json!({"type": "text"}));
            match fmt["type"].as_str() {
                Some("text") => {}
                Some("grammar") => {
                    check_grammar(&fmt).map_err(|e| bad(param.clone(), format!("Invalid custom tool grammar: {e}")))?
                }
                _ => return Err(bad(param, "Unsupported custom tool format.")),
            }
        }
        result.insert(native, (tool.clone(), namespace.map(str::to_owned)));
        Ok(())
    }
    let mut result = Specs::new();
    for (i, tool) in tools.iter().enumerate() {
        add(&mut result, tool, format!("{root}.{i}"), None, loaded)?;
    }
    Ok(result)
}

fn search_output_specs(item: &Value, param: &str) -> Result<Specs, ApiError> {
    validate_client_search_item(item, param)?;
    let Some(tools) = item["tools"].as_array() else {
        return Err(bad(format!("{param}.tools"), "Tool search output must contain a tools array."));
    };
    collect_specs(tools, &format!("{param}.tools"), true)
}

/// Callable tools, including those discovered by tool search in the history.
fn tool_specs(tools: &[Value], input: &Value) -> Result<Specs, ApiError> {
    let mut catalog = collect_specs(tools, "tools", false)?;
    let mut result: Specs = catalog
        .iter()
        .filter(|(_, (t, _))| !t["defer_loading"].as_bool().unwrap_or(false))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for (i, item) in input.as_array().into_iter().flatten().enumerate() {
        if item["type"] != "tool_search_output" {
            continue;
        }
        for (name, spec) in search_output_specs(item, &format!("input.{i}"))? {
            if let Some(previous) = catalog.get(&name) {
                let strip = |v: &Value| {
                    let mut v = v.clone();
                    if let Some(o) = v.as_object_mut() {
                        o.remove("defer_loading");
                    }
                    v
                };
                if strip(&previous.0) != strip(&spec.0) || previous.1 != spec.1 {
                    return Err(bad(format!("input.{i}.tools"), format!("Conflicting tool definition: {name}.")));
                }
            }
            catalog.insert(name.clone(), spec.clone());
            result.insert(name, spec);
        }
    }
    Ok(result)
}

fn chat_tools(specs: &Specs) -> Vec<Value> {
    specs
        .iter()
        .map(|(name, (tool, _))| {
            let mut description = tool["description"].as_str().unwrap_or_default().to_owned();
            let mut parameters = tool.get("parameters").cloned().unwrap_or(Value::Null);
            if tool["type"] == "custom" {
                description.push_str(
                    "\nPass the complete raw tool input as the string argument `input`. \
                     Do not wrap the string contents in another JSON object.",
                );
                if tool["format"]["type"] == "grammar" {
                    description.push_str(&format!(
                        "\nThe input must match this {} grammar:\n{}",
                        tool["format"]["syntax"].as_str().unwrap_or_default(),
                        tool["format"]["definition"].as_str().unwrap_or_default()
                    ));
                }
                parameters = json!({"type": "object", "properties": {"input": {"type": "string"}},
                                    "required": ["input"], "additionalProperties": false});
            }
            json!({"type": "function", "function": {"name": name, "description": description,
                   "parameters": parameters, "strict": tool.get("strict")}})
        })
        .collect()
}

/// A parsed chat tool call -> Responses output item.
fn output_tool_call(call_id: &str, name: &str, arguments: &str, specs: &Specs) -> Result<Value, String> {
    let Some((tool, namespace)) = specs.get(name) else {
        return Err(format!("Model returned an undeclared tool: {name}."));
    };
    let call_id = if call_id.is_empty() { format!("call_{}", rid()) } else { call_id.to_owned() };
    if tool["type"] == "tool_search" {
        let args: Value = serde_json::from_str(arguments).map_err(|_| "Tool search arguments must be a JSON object.".to_owned())?;
        if !args.is_object() {
            return Err("Tool search arguments must be a JSON object.".into());
        }
        return Ok(json!({"id": format!("ts_{}", rid()), "type": "tool_search_call", "call_id": call_id,
                         "execution": "client", "status": "completed", "arguments": args}));
    }
    let mut item = json!({"id": format!("fc_{}", rid()), "call_id": call_id, "name": tool["name"]});
    if let Some(ns) = namespace {
        item["namespace"] = json!(ns);
    }
    if tool["type"] == "custom" {
        let input = serde_json::from_str::<Value>(arguments)
            .ok()
            .and_then(|v| v["input"].as_str().map(str::to_owned))
            .ok_or_else(|| "Custom tool arguments must contain a string input.".to_owned())?;
        let fmt = &tool["format"];
        if fmt["type"] == "grammar" && fmt["syntax"] == "regex" {
            let re = regex::Regex::new(&format!("^(?:{})$", fmt["definition"].as_str().unwrap_or_default()))
                .map_err(|e| e.to_string())?;
            if !re.is_match(&input) {
                return Err(format!(
                    "Generated input does not match the grammar for {name}: regex mismatch (input length {} characters).",
                    input.chars().count()
                ));
            }
        }
        item["type"] = json!("custom_tool_call");
        item["input"] = json!(input);
    } else {
        item["type"] = json!("function_call");
        item["arguments"] = json!(arguments);
        item["status"] = json!("completed");
    }
    Ok(item)
}

// ---------------------------------------------------------------------------
// Input items -> chat messages
// ---------------------------------------------------------------------------

fn data_url_bytes(value: &str, param: &str) -> Result<(Vec<u8>, Option<String>), ApiError> {
    let (mime, payload, is_b64) = match value.strip_prefix("data:") {
        Some(rest) => {
            let (header, payload) = rest.split_once(',').ok_or_else(|| bad(param, "Invalid data URL in `file_data`."))?;
            let mut meta = header.split(';');
            let mime = meta.next().filter(|m| !m.is_empty()).map(str::to_owned);
            let b64 = meta.any(|m| m == "base64");
            (mime, payload, b64)
        }
        None => (None, value, true),
    };
    if !is_b64 {
        let decoded = percent_decode(payload);
        return Ok((decoded, mime));
    }
    base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map(|b| (b, mime))
        .map_err(|_| bad(param, "`file_data` must be Base64 or a data URL."))
}

fn percent_decode(s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Some(v) = std::str::from_utf8(&bytes[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok())
        {
            out.push(v);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    out
}

async fn download(url: &str, param: &str) -> Result<(Vec<u8>, Option<String>, Option<String>), ApiError> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(bad(param, "`file_url` must use http or https."));
    }
    let fail = |e: String| bad(param, format!("Unable to download `file_url`: {e}"));
    let resp = reqwest::Client::new()
        .get(url)
        .header("User-Agent", "gLLM/Responses")
        .timeout(Duration::from_secs(30))
        .send()
        .await
        .map_err(|e| fail(e.to_string()))?
        .error_for_status()
        .map_err(|e| fail(e.to_string()))?;
    if resp.content_length().is_some_and(|n| n as usize > MAX_INLINE_FILE_BYTES) {
        return Err(bad(param, "Input files are limited to 50 MB."));
    }
    let mime = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.split(';').next().unwrap_or(v).trim().to_owned());
    let name = url.split(['?', '#']).next().and_then(|u| u.rsplit('/').next()).filter(|n| !n.is_empty()).map(str::to_owned);
    let data = resp.bytes().await.map_err(|e| fail(e.to_string()))?;
    if data.len() > MAX_INLINE_FILE_BYTES {
        return Err(bad(param, "Input files are limited to 50 MB."));
    }
    Ok((data.to_vec(), mime, name))
}

fn guess_mime(name: &str) -> Option<&'static str> {
    let ext = name.rsplit_once('.')?.1.to_lowercase();
    Some(match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "pdf" => "application/pdf",
        "txt" => "text/plain",
        "csv" => "text/csv",
        "html" => "text/html",
        "md" => "text/markdown",
        "json" => "application/json",
        _ => return None,
    })
}

fn decode_text_file(data: &[u8], param: &str) -> Result<String, ApiError> {
    if data.starts_with(&[0xff, 0xfe]) || data.starts_with(&[0xfe, 0xff]) {
        let le = data.starts_with(&[0xff, 0xfe]);
        let units: Vec<u16> = data[2..]
            .as_chunks::<2>().0.iter()
            .map(|c| if le { u16::from_le_bytes([c[0], c[1]]) } else { u16::from_be_bytes([c[0], c[1]]) })
            .collect();
        if let Ok(s) = String::from_utf16(&units) {
            return Ok(s);
        }
    }
    let body = data.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(data);
    if let Ok(s) = std::str::from_utf8(body) {
        return Ok(s.to_owned());
    }
    if data[..data.len().min(4096)].contains(&0) {
        return Err(bad(param, "This binary document type is not supported yet."));
    }
    Ok(data.iter().map(|&b| b as char).collect()) // latin-1
}

/// Download `file_url` inputs so a stored request replays the same bytes.
async fn snapshot_files(input: &mut Value) -> Result<(), ApiError> {
    let Some(items) = input.as_array_mut() else { return Ok(()) };
    for (index, item) in items.iter_mut().enumerate() {
        let field = match item["type"].as_str() {
            None | Some("message") if item.get("role").is_some() => "content",
            Some("function_call_output") | Some("custom_tool_call_output") => "output",
            _ => continue,
        };
        let Some(parts) = item.get_mut(field).and_then(Value::as_array_mut) else { continue };
        for (pi, part) in parts.iter_mut().enumerate() {
            if part["type"] != "input_file" || part["file_url"].as_str().is_none_or(str::is_empty)
                || part.get("file_data").is_some_and(|v| !v.is_null()) || part.get("file_id").is_some_and(|v| !v.is_null())
            {
                continue;
            }
            let param = format!("input.{index}.{field}.{pi}");
            let url = part["file_url"].as_str().unwrap().to_owned();
            let (data, mime, name) = download(&url, &format!("{param}.file_url")).await?;
            let filename = part["filename"].as_str().map(str::to_owned).or(name).unwrap_or_else(|| "input_file".into());
            let mime = mime.or_else(|| guess_mime(&filename).map(str::to_owned)).unwrap_or_else(|| "application/octet-stream".into());
            let obj = part.as_object_mut().unwrap();
            obj.remove("file_url");
            obj.insert("filename".into(), json!(filename));
            obj.insert("file_data".into(), json!(format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(&data))));
        }
    }
    Ok(())
}

/// One `input_file` part -> chat content parts.
async fn input_file_parts(part: &Value, param: &str) -> Result<Vec<Value>, ApiError> {
    let sources: Vec<&str> = ["file_data", "file_url", "file_id"]
        .into_iter()
        .filter(|k| part[*k].as_str().is_some_and(|v| !v.is_empty()))
        .collect();
    if sources.len() != 1 {
        return Err(bad(param, "`input_file` requires exactly one of `file_data`, `file_url`, or `file_id`."));
    }
    let mut filename = part["filename"].as_str().map(str::to_owned);
    let (data, mut mime) = match sources[0] {
        "file_id" => {
            return Err(bad(format!("{param}.file_id"),
                "File IDs require the Files API, which this server does not implement yet; use `file_data` or `file_url`."));
        }
        "file_data" => data_url_bytes(part["file_data"].as_str().unwrap(), &format!("{param}.file_data"))?,
        _ => {
            let (data, mime, name) = download(part["file_url"].as_str().unwrap(), &format!("{param}.file_url")).await?;
            filename = filename.or(name);
            (data, mime)
        }
    };
    if data.len() > MAX_INLINE_FILE_BYTES {
        return Err(bad(param, "Input files are limited to 50 MB."));
    }
    let filename = filename.unwrap_or_else(|| "input_file".into());
    let ext = filename.rsplit_once('.').map(|(_, e)| e.to_lowercase()).unwrap_or_default();
    let mut m = mime.take().or_else(|| guess_mime(&filename).map(str::to_owned)).unwrap_or_else(|| "application/octet-stream".into());
    if m == "application/octet-stream" && data.starts_with(b"%PDF-") {
        m = "application/pdf".into();
    }
    if m.starts_with("image/") {
        let url = format!("data:{m};base64,{}", base64::engine::general_purpose::STANDARD.encode(&data));
        return Ok(vec![json!({"type": "image_url", "image_url": {"url": url}})]);
    }
    if m == "application/pdf" || ext == "pdf" {
        return Err(bad(param, "PDF file inputs are not supported by this server."));
    }
    if m.starts_with("text/") || TEXT_EXTENSIONS.contains(&ext.as_str()) {
        let text = decode_text_file(&data, param)?;
        return Ok(vec![json!({"type": "text", "text": format!("\n[File: {filename}]\n{text}\n")})]);
    }
    Err(bad(param, format!("Unsupported input file type {m:?}; use an image or a text/code/CSV file.")))
}

/// Responses content parts -> chat content (a string when there is no media).
async fn content_parts(content: &[Value], param: &str) -> Result<Value, ApiError> {
    let mut parts = Vec::new();
    let mut media = false;
    for (i, part) in content.iter().enumerate() {
        let p = format!("{param}.{i}");
        if !part.is_object() {
            return Err(bad(p, "Content parts must be objects."));
        }
        match part["type"].as_str() {
            Some("input_text") | Some("output_text") | Some("text") => {
                parts.push(json!({"type": "text", "text": part["text"].as_str().unwrap_or_default()}))
            }
            Some("input_image") => {
                let url = part["image_url"].as_str().filter(|u| !u.is_empty());
                let file_id = part["file_id"].as_str().filter(|u| !u.is_empty());
                if url.is_some() == file_id.is_some() {
                    return Err(bad(p, "`input_image` requires exactly one of `image_url` or `file_id`."));
                }
                if file_id.is_some() {
                    return Err(bad(format!("{p}.file_id"), "Image file IDs require the Files API; use `image_url` with a URL or data URL."));
                }
                parts.push(json!({"type": "image_url", "image_url": {"url": url.unwrap()}}));
                media = true;
            }
            Some("input_file") => {
                let file = input_file_parts(part, &p).await?;
                media |= file.iter().any(|f| f["type"] == "image_url");
                parts.extend(file);
            }
            Some("input_audio") => return Err(bad(p, "Audio input is not supported yet.")),
            other => return Err(bad(p, format!("Unsupported Responses content type: {other:?}."))),
        }
    }
    Ok(if media {
        Value::Array(parts)
    } else {
        Value::String(parts.iter().filter_map(|p| p["text"].as_str()).collect())
    })
}

async fn input_to_messages(instructions: &Value, input: &Value) -> Result<Vec<Value>, ApiError> {
    let mut messages: Vec<Value> = Vec::new();
    match instructions {
        Value::Null => {}
        Value::String(s) if !s.is_empty() => messages.push(json!({"role": "developer", "content": s})),
        Value::String(_) => {}
        _ => return Err(bad("instructions", "Only string instructions are supported.")),
    }
    if let Some(text) = input.as_str() {
        messages.push(json!({"role": "user", "content": text}));
        return Ok(messages);
    }
    for (index, item) in input.as_array().into_iter().flatten().enumerate() {
        let param = format!("input.{index}");
        if let Some(text) = item.as_str() {
            messages.push(json!({"role": "user", "content": text}));
            continue;
        }
        if !item.is_object() {
            return Err(bad(param, "Input items must be strings or objects."));
        }
        let mut item = item.clone();
        let mut kind = item["type"].as_str().map(str::to_owned);
        match kind.as_deref() {
            Some("tool_search_call") => {
                validate_client_search_item(&item, &param)?;
                if !item["arguments"].is_object() {
                    return Err(bad(format!("{param}.arguments"), "Tool search arguments must be an object."));
                }
                item = json!({"type": "function_call", "call_id": item["call_id"], "name": TOOL_SEARCH_NAME,
                              "arguments": item["arguments"].to_string()});
                kind = Some("function_call".into());
            }
            Some("tool_search_output") => {
                let discovered: Vec<String> = search_output_specs(&item, &param)?.into_keys().collect();
                messages.push(json!({"role": "tool", "tool_call_id": item["call_id"],
                                     "content": json!({"available_tools": discovered}).to_string()}));
                continue;
            }
            // Replayed thoughts are not user messages.
            Some("reasoning") => continue,
            _ => {}
        }
        match kind.as_deref() {
            Some("function_call") | Some("custom_tool_call") => {
                let call_id = item.get("call_id").filter(|v| !v.is_null()).or_else(|| item.get("id")).cloned().unwrap_or(Value::Null);
                let mut name = item["name"].as_str().unwrap_or_default().to_owned();
                if let Some(ns) = item["namespace"].as_str().filter(|n| !n.is_empty()) {
                    name = format!("{ns}.{name}");
                }
                let arguments = if kind.as_deref() == Some("custom_tool_call") {
                    let Some(input) = item["input"].as_str() else {
                        return Err(bad(param, "Custom tool input must be a string."));
                    };
                    json!({"input": input}).to_string()
                } else {
                    item["arguments"].as_str().unwrap_or("{}").to_owned()
                };
                let call = json!({"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}});
                // Keep text and calls of one assistant turn in one message.
                match messages.last_mut() {
                    Some(last) if last["role"] == "assistant" => {
                        if !last["tool_calls"].is_array() {
                            last["tool_calls"] = json!([]);
                        }
                        last["tool_calls"].as_array_mut().unwrap().push(call);
                    }
                    _ => messages.push(json!({"role": "assistant", "content": null, "tool_calls": [call]})),
                }
            }
            Some("function_call_output") | Some("custom_tool_call_output") => {
                let output = match &item["output"] {
                    Value::Array(parts) => content_parts(parts, &format!("{param}.output")).await?,
                    Value::String(s) => Value::String(s.clone()),
                    Value::Null => Value::String(String::new()),
                    other => Value::String(other.to_string()),
                };
                messages.push(json!({"role": "tool", "tool_call_id": item["call_id"], "content": output}));
            }
            None | Some("message") if item.get("role").is_some() => {
                let content = match &item["content"] {
                    Value::Array(parts) => content_parts(parts, &format!("{param}.content")).await?,
                    Value::String(s) => Value::String(s.clone()),
                    Value::Null => Value::String(String::new()),
                    _ => return Err(bad(format!("{param}.content"), "Message content must be text.")),
                };
                messages.push(json!({"role": item["role"], "content": content}));
            }
            other => return Err(bad(param, format!("Unsupported Responses input type: {other:?}."))),
        }
    }
    // Templates such as Qwen take instructions only first: merge them, keeping
    // the conversation order.
    let (instr, conversation): (Vec<Value>, Vec<Value>) =
        messages.into_iter().partition(|m| m["role"] == "system" || m["role"] == "developer");
    let mut messages = Vec::new();
    if !instr.is_empty() {
        let mut texts = Vec::new();
        for m in &instr {
            let Some(t) = m["content"].as_str() else {
                return Err(bad("input", "Instruction messages must contain text only."));
            };
            texts.push(t.to_owned());
        }
        messages.push(json!({"role": instr[0]["role"], "content": texts.join("\n\n")}));
    }
    messages.extend(conversation);
    if !messages.iter().any(|m| m["role"] == "user") {
        return Err(bad("input",
            "Responses requests must include a user message (either directly or via previous_response_id history)."));
    }
    Ok(messages)
}

fn previous_output_to_input(response: &Value) -> Vec<Value> {
    let mut items = Vec::new();
    for item in response["output"].as_array().into_iter().flatten() {
        match item["type"].as_str() {
            Some("message") if item["role"] == "assistant" => {
                items.push(json!({"type": "message", "role": "assistant", "content": item["content"]}))
            }
            Some(kind @ ("function_call" | "custom_tool_call")) => {
                let keys: &[&str] = if kind == "function_call" { &["type", "call_id", "name", "arguments"] } else { &["type", "call_id", "name", "input"] };
                let mut entry = Map::new();
                for k in keys {
                    entry.insert((*k).into(), item[*k].clone());
                }
                if item["namespace"].as_str().is_some_and(|n| !n.is_empty()) {
                    entry.insert("namespace".into(), item["namespace"].clone());
                }
                items.push(Value::Object(entry));
            }
            Some("tool_search_call") => items.push(item.clone()),
            _ => {}
        }
    }
    items
}

// ---------------------------------------------------------------------------
// Response objects
// ---------------------------------------------------------------------------

fn base_response(req: &Value, id: &str, created_at: u64, model: &str, has_tools: bool) -> Value {
    json!({
        "id": id, "object": "response", "created_at": created_at, "status": "in_progress",
        "background": false, "completed_at": null, "error": null, "incomplete_details": null,
        "instructions": req["instructions"], "max_output_tokens": req["max_output_tokens"],
        "max_tool_calls": req["max_tool_calls"], "model": model, "output": [],
        "parallel_tool_calls": req["parallel_tool_calls"].as_bool().unwrap_or(true),
        "previous_response_id": req["previous_response_id"],
        "reasoning": {"effort": req["reasoning"]["effort"], "summary": null},
        "store": req["store"].as_bool().unwrap_or(false), "temperature": req["temperature"],
        "text": if req["text"].is_null() { json!({"format": {"type": "text"}}) } else { req["text"].clone() },
        "tool_choice": if req["tool_choice"].is_null() { json!(if has_tools { "auto" } else { "none" }) } else { req["tool_choice"].clone() },
        "tools": if req["tools"].is_null() { json!([]) } else { req["tools"].clone() },
        "top_p": req["top_p"], "truncation": req["truncation"].as_str().unwrap_or("disabled"),
        "usage": null, "user": req["user"],
        "metadata": if req["metadata"].is_null() { json!({}) } else { req["metadata"].clone() },
        "service_tier": req["service_tier"],
    })
}

fn usage(prompt: usize, completion: usize) -> Value {
    json!({"input_tokens": prompt, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
           "output_tokens": completion, "output_tokens_details": {"reasoning_tokens": 0},
           "total_tokens": prompt + completion})
}

fn reasoning_item(text: &str, id: &str, status: &str) -> Value {
    json!({"id": id, "type": "reasoning", "status": status, "summary": [],
           "content": [{"type": "reasoning_text", "text": text}]})
}

fn message_item(id: &str, text: &str, status: &str) -> Value {
    json!({"id": id, "type": "message", "status": status, "role": "assistant",
           "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": text}]})
}

/// Reasoning state over one generation: started, and whether it ended.
#[derive(Default)]
struct Reasoning {
    started: bool,
    ended: bool,
}

fn completion_fields(finish: &FinishReason, reasoning: &Reasoning, has_output: bool, tool_error: Option<&str>) -> Value {
    let (mut status, mut error, mut incomplete) = ("completed", Value::Null, Value::Null);
    if matches!(finish, FinishReason::Length) {
        status = "incomplete";
        incomplete = json!({"reason": "max_output_tokens"});
    } else if let Some(e) = tool_error {
        status = "failed";
        error = json!({"code": "server_error", "message": e});
    } else if reasoning.started {
        if !reasoning.ended {
            error = json!({"code": "server_error", "message": "Generation ended before the reasoning block was closed."});
        } else if !has_output {
            error = json!({"code": "server_error", "message": "Generation ended after reasoning without an answer or tool call."});
        }
        if !error.is_null() {
            status = "failed";
        }
    }
    json!({"status": status, "completed_at": if status == "completed" { json!(unix_now()) } else { Value::Null },
           "error": error, "incomplete_details": incomplete})
}

fn merge(target: &mut Value, fields: Value) {
    if let (Some(t), Value::Object(f)) = (target.as_object_mut(), fields) {
        t.extend(f);
    }
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

fn check_capabilities(req: &Value) -> Result<(), ApiError> {
    let include_extra = req["include"]
        .as_array()
        .is_some_and(|v| v.iter().any(|i| i != "reasoning.encrypted_content"));
    for (cond, param) in [
        (req["background"] == true, "background"),
        (!req["conversation"].is_null(), "conversation"),
        (include_extra, "include"),
        (!req["max_tool_calls"].is_null(), "max_tool_calls"),
        (!req["moderation"].is_null(), "moderation"),
        (!req["prompt"].is_null(), "prompt"),
        (!req["prompt_cache_options"].is_null(), "prompt_cache_options"),
        (!req["top_logprobs"].is_null(), "top_logprobs"),
        (req["truncation"] == "auto", "truncation"),
    ] {
        if cond {
            return Err(ApiError::unsupported(param));
        }
    }
    match &req["tool_choice"] {
        Value::Null => Ok(()),
        Value::String(s) if s == "none" || s == "auto" => Ok(()),
        _ => Err(ApiError::invalid(
            "This runtime supports tool_choice='none' and 'auto' for Responses.",
            Some("tool_choice"),
            "unsupported_parameter",
        )),
    }
}

/// `text.format` -> a chat `response_format`.
fn response_format(text: &Value) -> Option<Value> {
    let fmt = &text["format"];
    match fmt["type"].as_str() {
        Some("json_schema") => Some(json!({"type": "json_schema", "json_schema": {
            "name": fmt["name"], "schema": fmt["schema"], "strict": fmt["strict"]}})),
        Some("json_object") => Some(json!({"type": "json_object"})),
        _ => None,
    }
}

pub async fn create_response(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    match run(state, body).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    }
}

async fn run(state: Arc<AppState>, body: Bytes) -> Result<Response, ApiError> {
    let mut req: Value = serde_json::from_slice(&body).map_err(|e| ApiError::invalid(e.to_string(), None, "invalid_request"))?;
    if !req.is_object() {
        return Err(ApiError::invalid("The request body must be an object.", None, "invalid_request"));
    }
    state.check_model(req["model"].as_str())?;
    if req["input"].is_null() {
        return Err(ApiError::invalid("Field required", Some("input"), "invalid_request"));
    }
    check_capabilities(&req)?;
    if let Some(n) = req["max_output_tokens"].as_i64()
        && n <= 0
    {
        return Err(ApiError::invalid("`max_output_tokens` must be positive.", Some("max_output_tokens"), "invalid_request"));
    }
    let store = req["store"].as_bool().unwrap_or(false);
    // What this request adds to the history.
    let current: Vec<Value> = match &req["input"] {
        Value::String(s) => vec![json!({"type": "message", "role": "user", "content": s})],
        Value::Array(items) => items.clone(),
        _ => return Err(bad("input", "Input must be a string or a list.")),
    };
    if let Some(prev_id) = req["previous_response_id"].as_str() {
        let Some(previous) = state.responses.get(prev_id) else {
            return Err(ApiError {
                status: StatusCode::NOT_FOUND,
                message: "The previous response was not found.".into(),
                kind: "invalid_request_error",
                param: Some("previous_response_id".into()),
                code: Some("response_not_found".into()),
            });
        };
        if previous["model"].as_str() != Some(state.public_model()) {
            return Err(ApiError::invalid("The previous response uses a different model.", Some("previous_response_id"), "invalid_request_error"));
        }
        let mut items: Vec<Value> = previous["input_items"].as_array().cloned().unwrap_or_default();
        items.extend(previous_output_to_input(&previous["response"]));
        items.extend(current.iter().cloned());
        req["input"] = Value::Array(items);
    }
    if store {
        snapshot_files(&mut req["input"]).await?;
    }
    let tools: Vec<Value> = req["tools"].as_array().cloned().unwrap_or_default();
    let specs = tool_specs(&tools, &req["input"])?;
    let messages = input_to_messages(&req["instructions"], &req["input"]).await?;
    let chat_tools = chat_tools(&specs);
    let mut chat = json!({
        "model": state.public_model(),
        "messages": messages,
        "max_completion_tokens": req["max_output_tokens"],
        "temperature": req["temperature"],
        "top_p": req["top_p"],
        "tools": if chat_tools.is_empty() { Value::Null } else { json!(chat_tools) },
        "tool_choice": req["tool_choice"],
        "parallel_tool_calls": req["parallel_tool_calls"],
        "reasoning_effort": req["reasoning"]["effort"],
        "stream": req["stream"].as_bool().unwrap_or(false),
    });
    if let Some(fmt) = response_format(&req["text"]) {
        chat["response_format"] = fmt;
    }
    let chat_req: ChatCompletionRequest =
        serde_json::from_value(chat).map_err(|e| ApiError::invalid(e.to_string(), None, "invalid_request"))?;
    let prepared = super::chat::prepare(&state, chat_req)?;
    let stream = prepared.stream;
    let events = state.chat.chat(prepared.request).await.map_err(submit_error)?;
    let ctx = Ctx {
        req,
        specs,
        id: format!("resp_{}", rid()),
        created_at: unix_now(),
        model: state.public_model().to_owned(),
        store: store.then(|| (state.clone(), current)),
    };
    if stream {
        Ok(Sse::new(stream_events(events, ctx)).keep_alive(KeepAlive::default()).into_response())
    } else {
        collect(events, ctx).await
    }
}

struct Ctx {
    req: Value,
    specs: Specs,
    id: String,
    created_at: u64,
    model: String,
    /// Set when the response is to be stored: (state, this request's input items).
    store: Option<(Arc<AppState>, Vec<Value>)>,
}

impl Ctx {
    fn base(&self) -> Value {
        base_response(&self.req, &self.id, self.created_at, &self.model, !self.specs.is_empty())
    }

    fn expose_reasoning(&self) -> bool {
        self.req["reasoning"]["summary"] != "none"
    }

    fn save(&self, response: &Value) {
        if let Some((state, input)) = &self.store {
            state.responses.put(
                self.id.clone(),
                json!({"response": response, "input_items": input, "model": self.model}),
            );
        }
    }
}

async fn collect(
    mut events: impl Stream<Item = vllm_chat::Result<ChatEvent>> + Unpin,
    ctx: Ctx,
) -> Result<Response, ApiError> {
    let mut reasoning = Reasoning::default();
    while let Some(event) = events.next().await {
        match event.map_err(submit_error)? {
            ChatEvent::BlockStart { kind: AssistantBlockKind::Reasoning, .. } => reasoning.started = true,
            ChatEvent::BlockEnd { block: AssistantContentBlock::Reasoning { .. }, .. } => reasoning.ended = true,
            ChatEvent::Done { message, usage: u, finish_reason, .. } => {
                if let Some(err) = finish_error(&finish_reason) {
                    return Err(err);
                }
                let mut text = String::new();
                let mut reasoning_text = String::new();
                let mut calls = Vec::new();
                for block in message.content {
                    match block {
                        AssistantContentBlock::Text { text: t } => text.push_str(&t),
                        AssistantContentBlock::Reasoning { text: t } => reasoning_text.push_str(&t),
                        AssistantContentBlock::ToolCall(c) => calls.push(c),
                    }
                }
                let mut response = ctx.base();
                let mut output = Vec::new();
                if !reasoning_text.is_empty() && ctx.expose_reasoning() {
                    let status = if reasoning.ended { "completed" } else { "incomplete" };
                    output.push(reasoning_item(&reasoning_text, &format!("rs_{}", rid()), status));
                }
                if !text.is_empty() || (calls.is_empty() && output.is_empty() && !reasoning.started) {
                    output.push(message_item(&format!("msg_{}", rid()), &text, "completed"));
                }
                let mut tool_error = None;
                for c in &calls {
                    match output_tool_call(&c.id, &c.name, &c.arguments, &ctx.specs) {
                        Ok(item) => output.push(item),
                        Err(e) => {
                            tool_error = Some(e);
                            break;
                        }
                    }
                }
                if let Some(e) = &tool_error {
                    return Err(ApiError { status: StatusCode::INTERNAL_SERVER_ERROR, message: e.clone(),
                                          kind: "server_error", param: None, code: Some("invalid_tool_output".into()) });
                }
                let has_output = !text.trim().is_empty() || !calls.is_empty();
                merge(&mut response, completion_fields(&finish_reason, &reasoning, has_output, None));
                response["output"] = json!(output);
                response["usage"] = usage(u.prompt_token_count, u.output_token_count);
                ctx.save(&response);
                return Ok(Json(response).into_response());
            }
            _ => {}
        }
    }
    Err(ApiError::server("response stream ended without a result"))
}

fn stream_events(
    mut events: impl Stream<Item = vllm_chat::Result<ChatEvent>> + Unpin + Send + 'static,
    ctx: Ctx,
) -> impl Stream<Item = Result<Event, Infallible>> + Send + 'static {
    async_stream::stream! {
        let mut seq = 0u64;
        let mut ev = |kind: &str, payload: Value| {
            let mut v = json!({"type": kind, "sequence_number": seq});
            merge(&mut v, payload);
            seq += 1;
            Event::default().event(kind).data(v.to_string())
        };
        let initial = ctx.base();
        yield Ok(ev("response.created", json!({"response": initial})));
        yield Ok(ev("response.in_progress", json!({"response": initial})));

        let expose = ctx.expose_reasoning();
        let mut outputs: Vec<Value> = Vec::new();
        let mut next_index = 0usize;
        let mut reasoning = Reasoning::default();
        let (mut rs_id, mut rs_index, mut rs_text, mut rs_done) = (None::<String>, 0usize, String::new(), false);
        let (mut msg_id, mut msg_index, mut msg_text) = (None::<String>, 0usize, String::new());
        let mut tool_error: Option<String> = None;
        let mut finish = FinishReason::Length;
        let mut usage_v = Value::Null;
        let mut has_calls = false;

        while let Some(event) = events.next().await {
            let event = match event {
                Ok(e) => e,
                Err(e) => {
                    let mut failed = initial.clone();
                    merge(&mut failed, json!({"status": "failed", "output": outputs,
                                              "error": {"code": "server_error", "message": submit_error(e).message}}));
                    yield Ok(ev("response.failed", json!({"response": failed})));
                    return;
                }
            };
            match event {
                ChatEvent::BlockStart { kind: AssistantBlockKind::Reasoning, .. } => reasoning.started = true,
                ChatEvent::BlockDelta { kind: AssistantBlockKind::Reasoning, delta, .. } if expose && !delta.is_empty() => {
                    if rs_id.is_none() {
                        let id = format!("rs_{}", rid());
                        rs_index = next_index;
                        next_index += 1;
                        yield Ok(ev("response.output_item.added", json!({"output_index": rs_index, "item": reasoning_item("", &id, "in_progress")})));
                        rs_id = Some(id);
                    }
                    rs_text.push_str(&delta);
                    yield Ok(ev("response.reasoning_text.delta", json!({"output_index": rs_index, "item_id": rs_id, "content_index": 0, "delta": delta})));
                }
                ChatEvent::BlockEnd { block: AssistantContentBlock::Reasoning { .. }, .. } => {
                    reasoning.ended = true;
                    if let Some(id) = &rs_id && !rs_done {
                        rs_done = true;
                        let item = reasoning_item(&rs_text, id, "completed");
                        outputs.push(item.clone());
                        yield Ok(ev("response.reasoning_text.done", json!({"output_index": rs_index, "item_id": id, "content_index": 0, "text": rs_text})));
                        yield Ok(ev("response.output_item.done", json!({"output_index": rs_index, "item": item})));
                    }
                }
                ChatEvent::BlockDelta { kind: AssistantBlockKind::Text, delta, .. } if !delta.is_empty() => {
                    if msg_id.is_none() {
                        let id = format!("msg_{}", rid());
                        msg_index = next_index;
                        next_index += 1;
                        yield Ok(ev("response.output_item.added", json!({"output_index": msg_index,
                            "item": {"id": id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}})));
                        yield Ok(ev("response.content_part.added", json!({"output_index": msg_index, "item_id": id, "content_index": 0,
                            "part": {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}})));
                        msg_id = Some(id);
                    }
                    msg_text.push_str(&delta);
                    yield Ok(ev("response.output_text.delta", json!({"output_index": msg_index, "item_id": msg_id, "content_index": 0, "delta": delta, "logprobs": []})));
                }
                ChatEvent::ToolCallEnd { call, .. } => {
                    // Text before a call is finished before the call is announced.
                    if let Some(id) = msg_id.take() {
                        let item = message_item(&id, &msg_text, "completed");
                        outputs.push(item.clone());
                        yield Ok(ev("response.output_text.done", json!({"output_index": msg_index, "item_id": id, "content_index": 0, "text": msg_text, "logprobs": []})));
                        yield Ok(ev("response.content_part.done", json!({"output_index": msg_index, "item_id": id, "content_index": 0, "part": item["content"][0]})));
                        yield Ok(ev("response.output_item.done", json!({"output_index": msg_index, "item": item})));
                        msg_text.clear();
                    }
                    let item = match output_tool_call(&call.id, &call.name, &call.arguments, &ctx.specs) {
                        Ok(item) => item,
                        Err(e) => {
                            let mut failed = initial.clone();
                            merge(&mut failed, json!({"status": "failed", "output": outputs, "error": {"code": "server_error", "message": e}}));
                            yield Ok(ev("response.failed", json!({"response": failed})));
                            return;
                        }
                    };
                    has_calls = true;
                    let index = next_index;
                    next_index += 1;
                    if item["type"] == "tool_search_call" {
                        let mut added = item.clone();
                        added["status"] = json!("in_progress");
                        added["arguments"] = json!({});
                        yield Ok(ev("response.output_item.added", json!({"output_index": index, "item": added})));
                        yield Ok(ev("response.output_item.done", json!({"output_index": index, "item": item})));
                        outputs.push(item);
                        continue;
                    }
                    let custom = item["type"] == "custom_tool_call";
                    let field = if custom { "input" } else { "arguments" };
                    let prefix = if custom { "response.custom_tool_call_input" } else { "response.function_call_arguments" };
                    let args = item[field].as_str().unwrap_or_default().to_owned();
                    let mut added = item.clone();
                    added[field] = json!("");
                    if !custom {
                        added["status"] = json!("in_progress");
                    }
                    yield Ok(ev("response.output_item.added", json!({"output_index": index, "item": added})));
                    if !args.is_empty() {
                        yield Ok(ev(&format!("{prefix}.delta"), json!({"output_index": index, "item_id": item["id"], "delta": args})));
                    }
                    let mut done = json!({"output_index": index, "item_id": item["id"], field: args});
                    if !custom {
                        done["name"] = item["name"].clone();
                    }
                    yield Ok(ev(&format!("{prefix}.done"), done));
                    yield Ok(ev("response.output_item.done", json!({"output_index": index, "item": item})));
                    outputs.push(item);
                }
                ChatEvent::Done { usage: u, finish_reason, .. } => {
                    if let Some(err) = finish_error(&finish_reason) {
                        tool_error = Some(err.message);
                    }
                    finish = finish_reason;
                    usage_v = usage(u.prompt_token_count, u.output_token_count);
                    break;
                }
                _ => {}
            }
        }
        if let Some(id) = &rs_id && !rs_done {
            let item = reasoning_item(&rs_text, id, "incomplete");
            outputs.push(item.clone());
            yield Ok(ev("response.reasoning_text.done", json!({"output_index": rs_index, "item_id": id, "content_index": 0, "text": rs_text})));
            yield Ok(ev("response.output_item.done", json!({"output_index": rs_index, "item": item})));
        }
        // Keep ordinary empty generations; never invent an answer after reasoning.
        if outputs.is_empty() && msg_id.is_none() && !reasoning.started {
            let id = format!("msg_{}", rid());
            msg_index = next_index;
            yield Ok(ev("response.output_item.added", json!({"output_index": msg_index,
                "item": {"id": id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}})));
            yield Ok(ev("response.content_part.added", json!({"output_index": msg_index, "item_id": id, "content_index": 0,
                "part": {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}})));
            msg_id = Some(id);
        }
        if let Some(id) = msg_id.take() {
            let item = message_item(&id, &msg_text, "completed");
            outputs.push(item.clone());
            yield Ok(ev("response.output_text.done", json!({"output_index": msg_index, "item_id": id, "content_index": 0, "text": msg_text, "logprobs": []})));
            yield Ok(ev("response.content_part.done", json!({"output_index": msg_index, "item_id": id, "content_index": 0, "part": item["content"][0]})));
            yield Ok(ev("response.output_item.done", json!({"output_index": msg_index, "item": item})));
        }
        let has_output = !msg_text.trim().is_empty() || has_calls
            || outputs.iter().any(|o| o["type"] == "message" && o["content"][0]["text"].as_str().is_some_and(|t| !t.trim().is_empty()));
        let mut final_response = initial.clone();
        merge(&mut final_response, completion_fields(&finish, &reasoning, has_output, tool_error.as_deref()));
        final_response["output"] = json!(outputs);
        final_response["usage"] = usage_v;
        ctx.save(&final_response);
        let kind = format!("response.{}", final_response["status"].as_str().unwrap_or("completed"));
        yield Ok(ev(&kind, json!({"response": final_response})));
    }
}
