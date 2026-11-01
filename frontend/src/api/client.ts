const API_BASE = "/api/v1";

interface FetchOptions extends RequestInit {
  params?: Record<string, string | number | boolean | undefined>;
}

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public code?: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function getAuthToken(): string | null {
  try {
    const stored = localStorage.getItem("auth");
    if (stored) {
      const { access_token } = JSON.parse(stored);
      return access_token || null;
    }
  } catch {
    return null;
  }
  return null;
}

export function setAuthTokens(tokens: {
  access_token: string;
  refresh_token: string;
}) {
  localStorage.setItem("auth", JSON.stringify(tokens));
}

export function clearAuthTokens() {
  localStorage.removeItem("auth");
}

export function getRefreshToken(): string | null {
  try {
    const stored = localStorage.getItem("auth");
    if (stored) {
      const { refresh_token } = JSON.parse(stored);
      return refresh_token || null;
    }
  } catch {
    return null;
  }
  return null;
}

async function refreshAccessToken(): Promise<string | null> {
  const refreshToken = getRefreshToken();
  if (!refreshToken) return null;

  try {
    const response = await fetch(`${API_BASE}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });

    if (!response.ok) {
      clearAuthTokens();
      return null;
    }

    const data = await response.json();
    setAuthTokens(data);
    return data.access_token;
  } catch {
    clearAuthTokens();
    return null;
  }
}

async function apiStream(
  endpoint: string,
  body: unknown,
  signal?: AbortSignal
): Promise<Response> {
  const headers = new Headers({ "Content-Type": "application/json" });
  const token = getAuthToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  const request = (requestHeaders: Headers) =>
    fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      headers: requestHeaders,
      body: JSON.stringify(body),
      signal,
    });

  let response = await request(headers);
  if (response.status === 401 && token) {
    const newToken = await refreshAccessToken();
    if (newToken) {
      headers.set("Authorization", `Bearer ${newToken}`);
      response = await request(headers);
    }
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new ApiError(
      response.status,
      errorData.detail || "Request failed",
      errorData.error_code
    );
  }
  if (!response.body) {
    throw new ApiError(response.status, "Streaming response body is missing");
  }
  return response;
}

export async function apiFetch<T>(
  endpoint: string,
  options: FetchOptions = {}
): Promise<T> {
  const { params, ...fetchOptions } = options;

  let url = `${API_BASE}${endpoint}`;
  if (params) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        searchParams.set(key, String(value));
      }
    });
    const qs = searchParams.toString();
    if (qs) url += `?${qs}`;
  }

  const headers: Record<string, string> = {
    ...(fetchOptions.headers as Record<string, string>),
  };

  if (!(fetchOptions.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const token = getAuthToken();
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const makeRequest = async (): Promise<T> => {
    const response = await fetch(url, {
      ...fetchOptions,
      headers,
    });

    if (response.status === 401 && token) {
      const newToken = await refreshAccessToken();
      if (newToken) {
        headers["Authorization"] = `Bearer ${newToken}`;
        const retryResponse = await fetch(url, {
          ...fetchOptions,
          headers,
        });
        if (!retryResponse.ok) {
          const errorData = await retryResponse.json().catch(() => ({}));
          throw new ApiError(
            retryResponse.status,
            errorData.detail || "Request failed",
            errorData.error_code
          );
        }
        return retryResponse.json();
      }
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new ApiError(
        response.status,
        errorData.detail || "Request failed",
        errorData.error_code
      );
    }

    return response.json();
  };

  return makeRequest();
}

export const api = {
  get: <T>(endpoint: string, options?: FetchOptions) =>
    apiFetch<T>(endpoint, { ...options, method: "GET" }),
  post: <T>(endpoint: string, body?: unknown, options?: FetchOptions) =>
    apiFetch<T>(endpoint, {
      ...options,
      method: "POST",
      body: body instanceof FormData ? body : JSON.stringify(body ?? {}),
    }),
  put: <T>(endpoint: string, body?: unknown, options?: FetchOptions) =>
    apiFetch<T>(endpoint, {
      ...options,
      method: "PUT",
      body: JSON.stringify(body ?? {}),
    }),
  patch: <T>(endpoint: string, body?: unknown, options?: FetchOptions) =>
    apiFetch<T>(endpoint, {
      ...options,
      method: "PATCH",
      body: JSON.stringify(body ?? {}),
    }),
  delete: <T>(endpoint: string, options?: FetchOptions) =>
    apiFetch<T>(endpoint, { ...options, method: "DELETE" }),
  stream: apiStream,
};
