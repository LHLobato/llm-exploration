# 08 — Segurança, Rate Limit e Observabilidade

## Segurança

- **Transporte**: HTTPS/HSTS obrigatório; CORS restrito às origens da SPA.
- **Segredos**: `service_role` e pepper só no backend; rotação documentada.
- **Entrada**: validação estrita (Pydantic) e normalização de domínio; limite de tamanho de corpo.
- **SSRF**: backend só faz DNS/WHOIS; resolvers de saída fixos; bloquear consultas a domínios internos (`.local`, `.internal`, sufixos privados) e a IPs privados.
- **Enumeração**: respostas de login/cadastro/reset genéricas; tempo de resposta equalizado.
- **Força bruta**: limites por IP e por email nas rotas `/auth/*` e de OTP.
- **Logs**: nunca registrar senha, OTP, chave completa ou JWT; mascarar email.
- **RLS ativo** em todas as tabelas públicas; testes automatizados de políticas.
- **Dependências**: `pip-audit`/`npm audit` e Dependabot no CI.
- **LGPD**: base legal, política de privacidade, exportação/exclusão de conta (cascata em `profiles`, `api_keys`, `api_requests`), retenção de logs definida.
- **Abuso**: o serviço consulta domínios potencialmente maliciosos de terceiros; não executar nem baixar conteúdo deles.

## Rate limiting e quotas

| Escopo | Sugestão inicial |
|--------|------------------|
| Por chave (`/v1/domains/analyze`) | 60 req/min, 5.000/dia |
| Por IP (sem credencial / auth) | 20 req/min |
| OTP envio | 1/60 s, 5/h por email |
| OTP verificação | 5 tentativas / código |

- Algoritmo: janela deslizante ou *token bucket*; armazenamento em Postgres (MVP) ou Redis.
- Respostas `429` com `Retry-After` e cabeçalhos `X-RateLimit-*`.
- Limites configuráveis por chave (coluna futura `plan`).

## Observabilidade

- Logs estruturados JSON com `request_id`, `user_id`, `api_key_id`, `channel`, latência, status.
- Métricas (Prometheus/OpenTelemetry): latência por etapa (DNS, WHOIS, atributos, imagem, inferência), taxa de `partial`, hits de cache, 4xx/5xx, uso por chave.
- Alertas: pico de 5xx, latência p95, falha de entrega de email OTP, taxa anômala de tentativas de login.
- Health: `/healthz` (liveness) e `/readyz` (BD + modelos carregados).

## Auditoria

`api_requests` (uso) e `otp_audit` (envios/tentativas) com retenção configurável.
