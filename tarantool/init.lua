box.cfg{ listen = 3301 }

-- Удаляем старое пространство (если нужно)
local cache_space = box.space.cache
if cache_space then
    cache_space:drop()
end

-- Создаём с явным форматом
box.schema.space.create('cache', {
    format = {
        {name = 'key',         type = 'string'},
        {name = 'value',       type = '*'},
        {name = 'expires_at',  type = 'number'}
    },
    if_not_exists = false  -- чтобы пересоздавалось
})

-- Первичный индекс: по key
box.space.cache:create_index('primary', {
    type = 'hash',
    parts = {'key'},
    if_not_exists = true
})

-- Вторичный индекс: по expires_at
box.space.cache:create_index('expires', {
    type = 'tree',
    parts = {'expires_at'},
    if_not_exists = true
})

print('✅ Tarantool: cache space created with correct format.')