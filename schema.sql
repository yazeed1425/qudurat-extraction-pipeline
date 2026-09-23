-- Qudurat RAG pipeline: Supabase schema
-- Run this in the Supabase SQL Editor.

create extension if not exists vector;
create extension if not exists pgcrypto;

drop function if exists match_qudurat_questions(vector, int, text);
drop table if exists "الجبر";

create table "الجبر" (
    id uuid primary key default gen_random_uuid(),
    source_page integer not null,
    source_pdf text not null default '',
    box_index integer,
    skill text not null,
    sub_skill text not null,
    difficulty text not null check (difficulty in ('easy', 'medium', 'hard')),
    question_text text not null,
    options jsonb not null,
    correct_answer text check (correct_answer in ('A', 'B', 'C', 'D')),
    explanation text default '',
    mathematically_verified boolean not null default false,
    confidence_score real not null default 0 check (confidence_score >= 0 and confidence_score <= 1),
    status text not null default 'pending_review'
        check (status in ('pending_review', 'approved', 'rejected')),
    has_image boolean not null default false,
    image_url text,  -- full box image, set only when has_image is true; see qudurat_pipeline/storage.py
    diagram_image_url text,  -- diagram-only crop, cropped from image_url via the model's reported bbox
    embedding vector(3072),  -- gemini-embedding-001; change to 1536 if you switch back to OpenAI
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

-- Storage bucket that holds the has_image=true question crops (public read,
-- written only by the pipeline's service_role key).
insert into storage.buckets (id, name, public)
values ('question-images', 'question-images', true)
on conflict (id) do nothing;

create index "جبر_embedding_idx"
    on "الجبر"
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

create index "جبر_skill_idx"
    on "الجبر" (skill, sub_skill);

create index "جبر_status_idx"
    on "الجبر" (status, mathematically_verified, confidence_score);

create index "جبر_source_idx"
    on "الجبر" (source_pdf, source_page, box_index);

-- Keep updated_at current whenever the review dashboard edits a row.
create or replace function set_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

create trigger "جبر_set_updated_at"
    before update on "الجبر"
    for each row
    execute function set_updated_at();

-- Similarity search helper for the RAG retrieval step. Only serves rows a human
-- has approved in the review dashboard, so unverified/rejected extractions never
-- leak into RAG answers.
create or replace function match_qudurat_questions(
    query_embedding vector(3072),
    match_count int default 5,
    filter_skill text default null
)
returns table (
    id uuid,
    source_page integer,
    skill text,
    sub_skill text,
    difficulty text,
    question_text text,
    options jsonb,
    correct_answer text,
    explanation text,
    mathematically_verified boolean,
    confidence_score real,
    has_image boolean,
    image_url text,
    diagram_image_url text,
    similarity float
)
language sql stable
as $$
    select
        q.id,
        q.source_page,
        q.skill,
        q.sub_skill,
        q.difficulty,
        q.question_text,
        q.options,
        q.correct_answer,
        q.explanation,
        q.mathematically_verified,
        q.confidence_score,
        q.has_image,
        q.image_url,
        q.diagram_image_url,
        1 - (q.embedding <=> query_embedding) as similarity
    from "الجبر" q
    where q.status = 'approved'
        and (filter_skill is null or q.skill = filter_skill)
    order by q.embedding <=> query_embedding
    limit match_count;
$$;
