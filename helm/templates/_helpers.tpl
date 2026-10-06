{{/* A digest takes precedence over a tag. Never combine both references. */}}
{{- define "energy-mlops.image" -}}
{{- $repository := required "image.repository is required" .repository -}}
{{- $digest := default "" .digest -}}
{{- if $digest -}}
  {{- if not (regexMatch "^sha256:[a-f0-9]{64}$" $digest) -}}
    {{- fail "image.digest must be sha256 followed by 64 lowercase hex characters" -}}
  {{- end -}}
  {{- printf "%s@%s" $repository $digest -}}
{{- else -}}
  {{- $tag := required "image.tag is required when image.digest is empty" .tag -}}
  {{- printf "%s:%s" $repository $tag -}}
{{- end -}}
{{- end -}}
