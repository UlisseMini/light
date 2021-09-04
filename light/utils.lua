local utils = {}

function utils.finally(fn, cleanup)
  local ret = table.pack(pcall(fn))
  cleanup()

  local status, err = table.unpack(ret)
  if not status then error(err) end

  table.remove(ret, 1)
  return table.unpack(ret)
end


function utils.concat(t1, t2)
  for _, v in ipairs(t2) do
    table.insert(t1, v)
  end
  return t1
end

function utils.number(t)
  if type(t) == 'number' then
    return t
  elseif type(t) == 'table' and type(t.data) == 'number' then
    return t.data
  else
    return nil
  end
end

function utils.walk(t, fn)
  if utils.number(t) then
    fn(t)
  else
    for _, v in ipairs(t) do
      utils.walk(v, fn)
    end
  end
end

function utils.flatten(data)
  local flattened = {}
  utils.walk(data, function(v) table.insert(flattened, v) end)
  return flattened
end

function utils.slice(t, a)
  local ret = {}
  for i=a,#t do
    ret[i - a + 1] = t[i]
  end
  return ret
end

function utils.pp(t, ident)
  if type(t) ~= 'table' then
    return tostring(t)
  else
    local s = '{'
    for _,v in ipairs(t) do
      s = s .. utils.pp(v) .. ', '
    end
    return s:sub(1, #s-2) .. '}'
  end
end

function utils.p(...)
  local s = ''
  for _, v in ipairs({...}) do
    s = s .. utils.pp(v) .. '\t'
  end
  print(s)
end

function utils.eq(a, b)
  if type(a) ~= 'table' or type(b) ~= 'table' then
    return a == b
  end
  if #a ~= #b then
    return false
  end

  for i=1,#a do
    if not utils.eq(a[i], b[i]) then
      return false
    end
  end
  return true
end

return utils
