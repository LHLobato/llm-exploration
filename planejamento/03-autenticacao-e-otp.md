# 03 — Autenticação e OTP (plataforma web)

A autenticação é da **plataforma web**. Os endpoints reais da API usam chave de API (doc 04).

## Cadastro

Campos: `username`, `first_name`, `last_name`, `birth_date`, `email`, `password` (+ confirmação no frontend). `full_name` é derivado.

Fluxo:
1. SPA envia `POST /auth/signup`.
2. Backend valida: formato/unicidade de `username`, email, política de senha (mín. 12 caracteres; checar contra lista de senhas vazadas se possível), `birth_date` plausível (não futura, não anterior a 1900).
3. Backend chama `supabase.auth.sign_up(email, password, options.data={username, first_name, last_name, birth_date})`.
4. Supabase envia o **OTP de confirmação por email** (template usando `{{ .Token }}`, não o link). Configurar "Confirm email" ativo.
5. SPA exibe tela de código; `POST /auth/signup/verify {email, token}` → `supabase.auth.verify_otp(type="signup")`.
6. Após verificação: criar/confirmar `profiles`, abrir sessão.
7. **Idade**: menor de 18 pode ter conta, mas **não recebe chave** (ver abaixo). Decisão alternativa em `10-pendencias`.

## Login (senha + OTP por email)

O Supabase não oferece nativamente "senha + OTP por email" como segundo fator; implementar em duas etapas no backend, **sem emitir sessão na etapa 1**:

1. `POST /auth/login {email, password}`
   - Validar a senha com um cliente Supabase descartável (`sign_in_with_password`) e **descartar a sessão retornada**.
   - Em sucesso: `sign_in_with_otp(email, should_create_user=false)` envia o código.
   - Resposta genérica: `{ "otp_required": true }`. Em erro de senha, mesma latência/mensagem genérica ("credenciais inválidas").
2. `POST /auth/login/verify {email, token}` → `verify_otp(type="email")` → devolve sessão (access + refresh).
3. Pós-login: SPA chama `GET /me` (perfil + estado da chave, ver doc 04).

Alternativa mais simples (se aceitável): OTP como único fator de login e senha só no cadastro. Registrada como decisão em aberto.

## OTP — regras

- 6 dígitos, validade curta (ex.: 10 min, configurável no Supabase), uso único.
- Limite de reenvio (ex.: 1 por 60 s, máx. 5/hora por email e por IP).
- Limite de tentativas de verificação (ex.: 5) com bloqueio temporário.
- OTP nunca é logado nem persistido no nosso banco.
- Template de email em `supabase/templates/` (PT-BR), sem links clicáveis para o código.
- Configurar SMTP próprio (Resend/SES) com SPF/DKIM/DMARC; o SMTP padrão do Supabase tem limite baixo.

## Sessão

- JWT do Supabase como bearer no acesso web; refresh token em cookie `HttpOnly; Secure; SameSite=Lax` quando o BFF intermediar, ou gerenciado pelo `supabase-js` com `persistSession`.
- Logout invalida o refresh token (`sign_out`).
- Reset de senha: `POST /auth/password/reset` → OTP por email → `verify_otp(type="recovery")` → `update_user(password)`.

## Verificação de idade (≥ 18)

- Calculada no servidor a partir de `birth_date` (data completa, fuso UTC).
- Aplicada em **dois pontos**: na rota de geração de chave e na função SQL de criação de chave (defesa em profundidade).
- Mensagem ao menor: conta criada, mas chave indisponível até completar 18 anos.
- `birth_date` não é editável livremente após o cadastro (alteração exige fluxo de suporte), para evitar burla.
- Nota de privacidade/LGPD: é dado pessoal; coletar só o necessário e informar a finalidade.

## Endpoints deste módulo

`POST /auth/signup`, `POST /auth/signup/verify`, `POST /auth/otp/resend`, `POST /auth/login`, `POST /auth/login/verify`, `POST /auth/logout`, `POST /auth/password/reset`, `POST /auth/password/reset/verify`. Contratos no doc 05.
