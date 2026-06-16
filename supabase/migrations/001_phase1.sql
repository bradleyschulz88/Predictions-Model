-- Phase 1: profiles, user settings, bets, subscriptions (Stripe-ready)
-- Run in Supabase SQL Editor or via supabase db push

create extension if not exists "pgcrypto";

create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  email text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.user_settings (
  user_id uuid primary key references auth.users (id) on delete cascade,
  starting_bankroll numeric not null default 0,
  odds_format text not null default 'decimal' check (odds_format in ('decimal', 'american')),
  updated_at timestamptz not null default now()
);

create table if not exists public.bets (
  id text not null,
  user_id uuid not null references auth.users (id) on delete cascade,
  payload jsonb not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (user_id, id)
);

create index if not exists bets_user_updated_idx on public.bets (user_id, updated_at desc);

create table if not exists public.subscriptions (
  user_id uuid primary key references auth.users (id) on delete cascade,
  stripe_customer_id text,
  stripe_subscription_id text,
  status text not null default 'inactive',
  current_period_end timestamptz,
  updated_at timestamptz not null default now()
);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, email)
  values (new.id, new.email)
  on conflict (id) do update set email = excluded.email, updated_at = now();

  insert into public.user_settings (user_id)
  values (new.id)
  on conflict (user_id) do nothing;

  insert into public.subscriptions (user_id, status)
  values (new.id, 'inactive')
  on conflict (user_id) do nothing;

  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

alter table public.profiles enable row level security;
alter table public.user_settings enable row level security;
alter table public.bets enable row level security;
alter table public.subscriptions enable row level security;

create policy "profiles_select_own" on public.profiles
  for select using (auth.uid() = id);

create policy "profiles_update_own" on public.profiles
  for update using (auth.uid() = id);

create policy "settings_select_own" on public.user_settings
  for select using (auth.uid() = user_id);

create policy "settings_upsert_own" on public.user_settings
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "bets_select_own" on public.bets
  for select using (auth.uid() = user_id);

create policy "bets_insert_own" on public.bets
  for insert with check (auth.uid() = user_id);

create policy "bets_update_own" on public.bets
  for update using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy "bets_delete_own" on public.bets
  for delete using (auth.uid() = user_id);

create policy "subscriptions_select_own" on public.subscriptions
  for select using (auth.uid() = user_id);
